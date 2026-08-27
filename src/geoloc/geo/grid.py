"""The global fusion grid.

Every soft constraint is evaluated onto a regular lat/lon lattice, then
combined in log space. The grid also carries two offline-derived lookups:
a country label per cell, and a habitation prior.

Country labelling is done by nearest-populated-place rather than by polygon
containment, because that keeps the tool dependency-free and air-gappable.
The trade-off is real and is documented for the analyst: labels within
~50 km of a land border, and over open water, should not be trusted. The
fused posterior itself is unaffected -- only the human-readable label is.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..config import SETTINGS

EARTH_RADIUS_KM = 6371.0088


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    cla = np.cos(la)
    return np.stack([cla * np.cos(lo), cla * np.sin(lo), np.sin(la)], axis=-1)


def chord_to_arc_km(chord: np.ndarray) -> np.ndarray:
    """Convert a unit-sphere chord length to great-circle kilometres."""
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class PlaceIndex:
    """KD-tree over GeoNames populated places, on the unit sphere."""

    tree: cKDTree
    lats: np.ndarray
    lons: np.ndarray
    names: list[str]
    countries: list[str]
    admin1: list[str]
    population: np.ndarray

    def nearest(self, lat: float, lon: float, k: int = 1):
        pt = latlon_to_xyz(np.array([lat]), np.array([lon]))
        chord, idx = self.tree.query(pt, k=k)
        chord = np.atleast_2d(chord)[0]
        idx = np.atleast_2d(idx)[0]
        return [
            {
                "name": self.names[i],
                "country": self.countries[i],
                "admin1": self.admin1[i],
                "lat": float(self.lats[i]),
                "lon": float(self.lons[i]),
                "population": int(self.population[i]),
                "distance_km": float(chord_to_arc_km(np.array(c))),
            }
            for c, i in zip(chord, idx, strict=True)
        ]


@functools.lru_cache(maxsize=1)
def place_index() -> PlaceIndex:
    import geonamescache

    # Pinned explicitly rather than left to the default. geonamescache ships
    # four city datasets totalling ~190 MB and loads whichever matches this
    # threshold; the packaged app excludes the other three, so changing this
    # number would break the frozen build.
    gc = geonamescache.GeonamesCache(min_city_population=15000)
    cities = gc.get_cities()
    lats, lons, names, countries, admin1, pops = [], [], [], [], [], []
    for c in cities.values():
        try:
            lats.append(float(c["latitude"]))
            lons.append(float(c["longitude"]))
        except (KeyError, TypeError, ValueError):
            continue
        names.append(c.get("name", ""))
        countries.append(c.get("countrycode", ""))
        admin1.append(str(c.get("admin1code", "")))
        pops.append(int(c.get("population", 0) or 0))

    lat_a = np.asarray(lats, dtype=np.float64)
    lon_a = np.asarray(lons, dtype=np.float64)
    tree = cKDTree(latlon_to_xyz(lat_a, lon_a))
    return PlaceIndex(
        tree=tree, lats=lat_a, lons=lon_a, names=names,
        countries=countries, admin1=admin1,
        population=np.asarray(pops, dtype=np.int64),
    )


@dataclass
class WorldGrid:
    step: float
    lat_centers: np.ndarray   # (H,)
    lon_centers: np.ndarray   # (W,)
    lat2d: np.ndarray         # (H, W)
    lon2d: np.ndarray         # (H, W)
    country_idx: np.ndarray   # (H, W) int32, index into `countries`
    countries: list[str]
    dist_km: np.ndarray       # (H, W) distance to nearest populated place
    cell_area: np.ndarray     # (H, W) relative area, cos(lat) weighted

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat2d.shape

    def country_mask(self, iso2: str) -> np.ndarray:
        try:
            i = self.countries.index(iso2)
        except ValueError:
            return np.zeros(self.shape, dtype=bool)
        return self.country_idx == i

    def habitation_log_prior(self, scale_km: float = 400.0, floor: float = -6.0) -> np.ndarray:
        """Soft prior favouring cells near populated places.

        This is what keeps the posterior off the open ocean without needing a
        coastline polygon set. It is deliberately gentle: a genuine wilderness
        photo is penalised but never excluded.
        """
        return np.clip(-self.dist_km / scale_km, floor, 0.0)

    def cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        r = int(np.clip((lat + 90.0) / self.step, 0, self.shape[0] - 1))
        c = int(np.clip((lon + 180.0) / self.step, 0, self.shape[1] - 1))
        return r, c


def _cache_path(step: float):
    return SETTINGS.model_cache / f"worldgrid_{step:g}.npz"


@functools.lru_cache(maxsize=4)
def world_grid(step: float | None = None) -> WorldGrid:
    """Build (or load) the fusion grid.

    Construction is a 250k-point KD-tree query and takes ~10s, so the result
    is memoised in-process and cached to disk keyed on the step size.
    """
    step = step or SETTINGS.grid_step
    cache = _cache_path(step)
    if cache.exists():
        try:
            return _load_cached(cache, step)
        except Exception:
            cache.unlink(missing_ok=True)  # stale/corrupt cache, rebuild
    grid = _build_grid(step)
    try:
        SETTINGS.model_cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, step=step, lat_centers=grid.lat_centers,
            lon_centers=grid.lon_centers, country_idx=grid.country_idx,
            countries=np.array(grid.countries), dist_km=grid.dist_km.astype(np.float32),
        )
    except OSError:
        pass  # cache is an optimisation, never a requirement
    return grid


def _load_cached(path, step: float) -> WorldGrid:
    z = np.load(path, allow_pickle=False)
    lat_centers = z["lat_centers"]
    lon_centers = z["lon_centers"]
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    area = np.cos(np.radians(lat2d))
    area = area / area.sum()
    return WorldGrid(
        step=step, lat_centers=lat_centers, lon_centers=lon_centers,
        lat2d=lat2d, lon2d=lon2d,
        country_idx=z["country_idx"], countries=[str(c) for c in z["countries"]],
        dist_km=z["dist_km"].astype(np.float64), cell_area=area,
    )


def _build_grid(step: float) -> WorldGrid:
    lat_centers = np.arange(-90.0 + step / 2, 90.0, step)
    lon_centers = np.arange(-180.0 + step / 2, 180.0, step)
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

    idx = place_index()
    chord, nn = idx.tree.query(latlon_to_xyz(lat2d.ravel(), lon2d.ravel()), k=1)
    dist = chord_to_arc_km(chord).reshape(lat2d.shape)

    uniq = sorted({c for c in idx.countries if c})
    lut = {c: i for i, c in enumerate(uniq)}
    cc = np.array([lut.get(idx.countries[i], -1) for i in nn], dtype=np.int32)

    area = np.cos(np.radians(lat2d))
    area = area / area.sum()

    return WorldGrid(
        step=step, lat_centers=lat_centers, lon_centers=lon_centers,
        lat2d=lat2d, lon2d=lon2d,
        country_idx=cc.reshape(lat2d.shape), countries=uniq,
        dist_km=dist, cell_area=area,
    )
