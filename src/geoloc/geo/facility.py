"""Facility-type inference, and the site gazetteer it unlocks.

The insight this implements: words like "LIBRARY" or "DEPARTURES" are
worthless for geocoding -- querying a gazetteer for "LIBRARY" returns a
library, just not the right one -- but they are excellent evidence of what
*kind* of place the photograph shows. Combined with a country, that converts
into something factual and useful: the actual coordinates of every facility
of that kind in the country.

That is the difference between "somewhere in Australia" (7.7 million km2)
and "one of these 346 Australian university campuses", which is a list an
analyst can work through. It relies only on capabilities that measure well:
country estimation from the scene model, text from OCR, and OSM as a
gazetteer. It deliberately does not ask a model to name the place, which
does not work -- see the README.
"""
from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import SETTINGS, OfflineError


@dataclass(frozen=True)
class Facility:
    key: str
    label: str
    keywords: tuple[str, ...]
    selectors: tuple[str, ...]   # OSM tag filters
    typical_extent_km: float = 1.0


# Ordered by how strongly the keywords imply the facility.
FACILITIES: tuple[Facility, ...] = (
    Facility("university", "university or college campus",
             ("library", "faculty", "campus", "lecture", "lecturer", "student",
              "students", "university", "college", "seminar", "tutorial",
              "undergraduate", "postgraduate", "refectory", "union building",
              "school of", "department of", "halls of residence"),
             ('["amenity"="university"]', '["amenity"="college"]'), 1.2),
    Facility("hospital", "hospital",
             ("emergency", "outpatients", "ward", "radiology", "maternity",
              "surgery", "clinic", "hospital", "casualty", "triage"),
             ('["amenity"="hospital"]',), 0.6),
    Facility("airport", "airport",
             ("departures", "arrivals", "baggage", "terminal", "gate",
              "check-in", "boarding", "duty free", "airport"),
             ('["aeroway"="aerodrome"]',), 3.0),
    Facility("railway", "railway station",
             ("platform", "ticket office", "railway", "station", "trains",
              "timetable", "concourse"),
             ('["railway"="station"]',), 0.5),
    Facility("stadium", "stadium",
             ("stadium", "grandstand", "turnstile", "north stand",
              "south stand", "east stand", "west stand"),
             ('["leisure"="stadium"]',), 0.5),
    Facility("museum", "museum or gallery",
             ("museum", "gallery", "exhibition", "curator", "collection"),
             ('["tourism"="museum"]', '["tourism"="gallery"]'), 0.4),
    Facility("marina", "marina or harbour",
             ("marina", "harbour", "harbor", "yacht", "berth", "jetty",
              "ferry terminal"),
             ('["leisure"="marina"]', '["amenity"="ferry_terminal"]'), 0.8),
    Facility("place_of_worship", "place of worship",
             ("cathedral", "chapel", "parish", "mosque", "synagogue",
              "temple", "abbey", "minster"),
             ('["amenity"="place_of_worship"]',), 0.2),
    Facility("prison", "prison",
             ("correctional", "penitentiary", "remand", "hm prison"),
             ('["amenity"="prison"]',), 0.6),
    Facility("shopping", "shopping centre",
             ("food court", "shopping centre", "shopping center", "mall",
              "level 1", "level 2", "car park level"),
             ('["shop"="mall"]', '["shop"="department_store"]'), 0.4),
)

_WORD = re.compile(r"[a-z']+")
CACHE_TTL_SECONDS = 60 * 60 * 24 * 30  # a month; campuses do not move


def infer_facilities(texts: list[str]) -> list[tuple[Facility, list[str]]]:
    """Facilities implied by the text, best first, with the matching words."""
    corpus = " ".join(t.lower() for t in texts)
    words = set(_WORD.findall(corpus))

    hits: list[tuple[Facility, list[str], int]] = []
    for fac in FACILITIES:
        matched = []
        for kw in fac.keywords:
            if " " in kw:
                if kw in corpus:
                    matched.append(kw)
            elif kw in words:
                matched.append(kw)
        if matched:
            hits.append((fac, matched, len(matched)))

    hits.sort(key=lambda h: -h[2])
    return [(f, m) for f, m, _ in hits]


def _cache_path(iso2: str, key: str):
    d = SETTINGS.model_cache / "gazetteer"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{iso2.upper()}_{key}.json"


def _load_cache(iso2: str, key: str) -> list[dict[str, Any]] | None:
    path = _cache_path(iso2, key)
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - blob.get("fetched_at", 0) > CACHE_TTL_SECONDS:
        return None
    return blob.get("sites")


def _save_cache(iso2: str, key: str, sites: list[dict[str, Any]]) -> None:
    # The cache is an optimisation; failing to write it must never break a run.
    with contextlib.suppress(OSError):
        _cache_path(iso2, key).write_text(
            json.dumps({"fetched_at": time.time(), "sites": sites}))


def _build_query(iso2: str, fac: Facility, limit: int) -> str:
    parts = []
    for sel in fac.selectors:
        for kind in ("node", "way", "relation"):
            parts.append(f'  {kind}{sel}(area.a);')
    body = "\n".join(parts)
    return (f'[out:json][timeout:90];\n'
            f'area["ISO3166-1"="{iso2.upper()}"][admin_level=2]->.a;\n'
            f'(\n{body}\n);\nout center {limit};')


def fetch_sites(iso2: str, fac: Facility, *, limit: int = 1500,
                use_cache: bool = True) -> list[dict[str, Any]]:
    """Every site of this facility type in a country, from OSM.

    Cached on disk: the query takes ~10 s against a busy public endpoint and
    the answer is stable for months.
    """
    if use_cache:
        cached = _load_cache(iso2, fac.key)
        if cached is not None:
            return cached

    if not SETTINGS.net_allowed():
        raise OfflineError(
            f"Cannot fetch the {fac.label} gazetteer for {iso2}: offline mode "
            "is on and nothing is cached for this country yet.")

    query = _build_query(iso2, fac, limit)
    from .overpass import _overpass_endpoints

    data = None
    for endpoint in _overpass_endpoints():
        try:
            with httpx.Client(timeout=120,
                              headers={"User-Agent": SETTINGS.user_agent}) as c:
                resp = c.post(endpoint, data={"data": query})
            if resp.status_code in (429, 502, 503, 504):
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (httpx.HTTPError, ValueError):
            continue
    if data is None:
        raise RuntimeError(f"no Overpass endpoint answered the {fac.label} query")

    sites: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for el in data.get("elements", []):
        centre = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if centre.get("lat") is None:
            continue
        lat, lon = float(centre["lat"]), float(centre["lon"])
        # Campuses are mapped as overlapping nodes, ways and relations; round
        # to ~100 m so one site does not contribute a dozen near-identical
        # points and skew the posterior toward whichever place is mapped most.
        rounded = (round(lat, 3), round(lon, 3))
        if rounded in seen:
            continue
        seen.add(rounded)
        tags = el.get("tags", {})
        sites.append({
            "name": tags.get("name", ""),
            "lat": lat, "lon": lon,
            "osm": f"https://www.openstreetmap.org/{el.get('type')}/{el.get('id')}",
        })

    _save_cache(iso2, fac.key, sites)
    return sites
