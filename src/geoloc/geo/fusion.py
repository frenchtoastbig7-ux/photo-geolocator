"""Evidence fusion.

Each `GeoConstraint` is rendered to a log-likelihood field over the world
grid; fields are summed with per-evidence confidence weights and a habitation
prior, then normalised to a posterior.

Design notes that matter for correctness of the *reasoning*, not just the code:

* Every soft constraint has a non-zero floor. A single mistaken CLIP call
  must not be able to drive the true location to probability zero -- that is
  the failure mode that makes automated geolocation confidently wrong.
* `CERTAIN` constraints (an intact EXIF GPS tag) dominate the posterior, as
  they should -- but a GPS tag that disagrees with the scene content is a
  finding in its own right (a forged tag, a photo of a screen, a mislabelled
  file). Domination would bury it, so contradictions are detected separately
  by `detect_contradictions` and reported to the analyst rather than being
  left to show up as a shape in the posterior.
* Entropy of the final posterior is reported. A top candidate drawn from a
  near-uniform posterior is noise, and the analyst is told so.
"""
from __future__ import annotations

import numpy as np

from ..models import Candidate, Confidence, ConstraintKind, Evidence, GeoConstraint
from .grid import WorldGrid, place_index, world_grid

LOG_FLOOR = -20.0


def _soft_band(values: np.ndarray, lo: float, hi: float, softness: float) -> np.ndarray:
    """1.0 inside [lo, hi], Gaussian falloff outside over `softness` degrees."""
    below = np.clip(lo - values, 0.0, None)
    above = np.clip(values - hi, 0.0, None)
    excess = below + above
    s = max(softness, 1e-6)
    return np.exp(-0.5 * (excess / s) ** 2)


def render_constraint(c: GeoConstraint, grid: WorldGrid,
                      heatmaps: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Return a log-likelihood field of grid shape for one constraint."""
    H, W = grid.shape
    heatmaps = heatmaps or {}

    if c.kind is ConstraintKind.POINT:
        if c.lat is None or c.lon is None:
            return np.zeros((H, W))
        # The Gaussian is sampled at cell centres, so a radius much tighter
        # than a cell underflows to the floor at *every* cell and the
        # constraint silently vanishes -- which would let a coarse grid
        # discard the strongest evidence in the case. Widen to at least one
        # cell; exact coordinates are recovered separately in
        # `extract_candidates`, so no precision is lost in the output.
        cell_km = grid.step * 111.0
        radius = max(c.radius_km or 5.0, 0.5, cell_km)
        # Great-circle distance from the point to every cell.
        la1, la2 = np.radians(c.lat), np.radians(grid.lat2d)
        dl = np.radians(grid.lon2d - c.lon)
        a = (np.sin((la2 - la1) / 2) ** 2
             + np.cos(la1) * np.cos(la2) * np.sin(dl / 2) ** 2)
        dist = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        lik = np.exp(-0.5 * (dist / radius) ** 2)
        return np.log(np.maximum(lik, np.exp(LOG_FLOOR)))

    if c.kind is ConstraintKind.COUNTRIES:
        scores = c.country_scores or {}
        field = np.full((H, W), float(c.floor))
        if scores:
            hi = max(scores.values()) or 1.0
            for iso2, s in scores.items():
                m = grid.country_mask(iso2)
                if m.any():
                    field[m] = np.maximum(field[m], max(s / hi, c.floor))
        return np.log(np.maximum(field, np.exp(LOG_FLOOR)))

    if c.kind is ConstraintKind.LAT_BAND:
        lo = -90.0 if c.lat_min is None else c.lat_min
        hi = 90.0 if c.lat_max is None else c.lat_max
        lik = _soft_band(grid.lat2d, lo, hi, c.softness_deg)
        return np.log(np.maximum(lik, max(c.floor, np.exp(LOG_FLOOR))))

    if c.kind is ConstraintKind.LON_BAND:
        lo = -180.0 if c.lon_min is None else c.lon_min
        hi = 180.0 if c.lon_max is None else c.lon_max
        lik = _soft_band(grid.lon2d, lo, hi, c.softness_deg)
        return np.log(np.maximum(lik, max(c.floor, np.exp(LOG_FLOOR))))

    if c.kind is ConstraintKind.BBOX:
        lat_l = _soft_band(grid.lat2d, c.lat_min if c.lat_min is not None else -90,
                           c.lat_max if c.lat_max is not None else 90, c.softness_deg)
        lon_l = _soft_band(grid.lon2d, c.lon_min if c.lon_min is not None else -180,
                           c.lon_max if c.lon_max is not None else 180, c.softness_deg)
        lik = lat_l * lon_l
        return np.log(np.maximum(lik, max(c.floor, np.exp(LOG_FLOOR))))

    if c.kind is ConstraintKind.HEATMAP:
        hm = heatmaps.get(c.heatmap_ref or "")
        if hm is None or hm.shape != (H, W):
            return np.zeros((H, W))
        return np.log(np.maximum(hm / (hm.max() or 1.0), np.exp(LOG_FLOOR)))

    return np.zeros((H, W))


def fuse(evidence: list[Evidence], *, grid: WorldGrid | None = None,
         heatmaps: dict[str, np.ndarray] | None = None,
         use_habitation_prior: bool = True) -> tuple[np.ndarray, WorldGrid]:
    """Combine all evidence into a normalised posterior over the grid."""
    grid = grid or world_grid()
    logp = np.zeros(grid.shape)

    if use_habitation_prior:
        logp += grid.habitation_log_prior()

    for ev in evidence:
        for c in ev.constraints:
            w = (c.confidence or ev.confidence).weight
            logp += w * render_constraint(c, grid, heatmaps)

    logp -= logp.max()
    post = np.exp(logp) * grid.cell_area
    total = post.sum()
    if total <= 0 or not np.isfinite(total):
        post = grid.cell_area.copy()
        total = post.sum()
    return post / total, grid


def posterior_entropy_bits(post: np.ndarray) -> float:
    p = post[post > 0]
    return float(-(p * np.log2(p)).sum())


def uniform_entropy_bits(grid: WorldGrid) -> float:
    a = grid.cell_area[grid.cell_area > 0]
    return float(-(a * np.log2(a)).sum())


def top_countries(post: np.ndarray, grid: WorldGrid, n: int = 8) -> list[tuple[str, float]]:
    """Aggregate posterior mass per country label."""
    flat_c = grid.country_idx.ravel()
    flat_p = post.ravel()
    valid = flat_c >= 0
    if not valid.any():
        return []
    sums = np.bincount(flat_c[valid], weights=flat_p[valid],
                       minlength=len(grid.countries))
    order = np.argsort(sums)[::-1][:n]
    return [(grid.countries[i], float(sums[i])) for i in order if sums[i] > 0]


def _point_anchors(evidence: list[Evidence]) -> list[tuple[float, float]]:
    """Exact coordinates asserted by high-confidence point constraints."""
    out = []
    for ev in evidence:
        for c in ev.constraints:
            if (c.kind is ConstraintKind.POINT and c.lat is not None
                    and c.lon is not None
                    and (c.confidence or ev.confidence) in
                    (Confidence.CERTAIN, Confidence.HIGH)):
                out.append((float(c.lat), float(c.lon)))
    return out


def _anchor_in_cell(anchors: list[tuple[float, float]], grid: WorldGrid,
                    r: int, c: int) -> tuple[float, float] | None:
    """Return the anchor falling inside grid cell (r, c), if any."""
    for lat, lon in anchors:
        if grid.cell_of(lat, lon) == (r, c):
            return lat, lon
    return None


def extract_candidates(post: np.ndarray, grid: WorldGrid, *, n: int = 8,
                       min_separation_km: float = 120.0,
                       min_relative_score: float = 0.02,
                       evidence: list[Evidence] | None = None) -> list[Candidate]:
    """Greedy non-maximum suppression over the posterior.

    Suppression radius stops the top-N degenerating into eight adjacent cells
    of one blob, which would give the analyst a false sense of agreement.

    Candidates carrying less than `min_relative_score` of the leader's mass
    are dropped rather than padded out to `n`. Listing near-zero hypotheses
    alongside real ones invites them to be read as leads; a short list is the
    honest output when the evidence supports only a few places.
    """
    idx = place_index()
    work = post.copy()
    chosen: list[Candidate] = []
    ev_ids = [e.id for e in (evidence or []) if e.constraints]
    peak = float(post.max())

    # Exact coordinates from strong point evidence (an EXIF GPS tag, a
    # geocoded street name) are reported at full precision rather than
    # snapped to a cell centre, which on a 0.5 deg grid would move the
    # answer by up to ~40 km.
    anchors = _point_anchors(evidence or [])

    for rank in range(1, n + 1):
        flat = int(np.argmax(work))
        r, c = divmod(flat, grid.shape[1])
        score = float(post[r, c])
        if score <= 0 or (peak > 0 and score / peak < min_relative_score):
            break
        lat = float(grid.lat2d[r, c])
        lon = float(grid.lon2d[r, c])

        anchor = _anchor_in_cell(anchors, grid, r, c)
        if anchor is not None:
            lat, lon = anchor

        near = idx.nearest(lat, lon, k=1)
        place = near[0] if near else {}
        cc = grid.country_idx[r, c]
        country = grid.countries[cc] if cc >= 0 else ""

        chosen.append(Candidate(
            rank=rank, lat=lat, lon=lon, score=score,
            label=f"{place.get('name', 'unknown')}, {country}".strip(", "),
            country=country,
            admin1=str(place.get("admin1", "")),
            nearest_place=str(place.get("name", "")),
            distance_to_place_km=place.get("distance_km"),
            supporting_evidence=ev_ids,
        ))

        # Zero out a neighbourhood so the next pick is a distinct hypothesis.
        dlat = min_separation_km / 111.0
        dlon = min_separation_km / max(111.0 * np.cos(np.radians(lat)), 1.0)
        rlo, rhi = grid.cell_of(lat - dlat, lon)[0], grid.cell_of(lat + dlat, lon)[0]
        clo, chi = grid.cell_of(lat, lon - dlon)[1], grid.cell_of(lat, lon + dlon)[1]
        work[min(rlo, rhi):max(rlo, rhi) + 1, min(clo, chi):max(clo, chi) + 1] = 0.0

    return chosen


def detect_contradictions(evidence: list[Evidence], grid: WorldGrid,
                          heatmaps: dict[str, np.ndarray] | None = None,
                          *, min_confidence: Confidence = Confidence.HIGH,
                          agreement_threshold: float = 0.02) -> list[str]:
    """Report pairs of strong evidence whose supported regions barely overlap.

    Strong evidence pointing at disjoint parts of the world is one of the
    more informative things an analyst can learn: it usually means a forged
    or inherited GPS tag, a screenshot of someone else's photo, or a
    misattributed file. Because a `CERTAIN` constraint legitimately dominates
    the posterior, such a conflict is invisible in the fused output -- so it
    is detected here on the individual likelihood fields instead.

    Agreement is the cosine similarity of the two normalised likelihood
    fields: 1.0 is identical support, 0.0 is disjoint support.
    """
    order = [Confidence.SPECULATIVE, Confidence.LOW, Confidence.MEDIUM,
             Confidence.HIGH, Confidence.CERTAIN]
    floor = order.index(min_confidence)

    fields: list[tuple[str, np.ndarray]] = []
    for ev in evidence:
        for c in ev.constraints:
            conf = c.confidence or ev.confidence
            if order.index(conf) < floor:
                continue
            f = np.exp(render_constraint(c, grid, heatmaps))
            total = f.sum()
            if total > 0 and np.isfinite(total):
                fields.append((ev.title, f / total))

    warnings: list[str] = []
    for i, (name_a, fa) in enumerate(fields):
        for name_b, fb in fields[i + 1:]:
            denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
            if denom <= 0:
                continue
            agreement = float((fa * fb).sum() / denom)
            if agreement < agreement_threshold:
                warnings.append(
                    f"Contradictory evidence: '{name_a}' and '{name_b}' "
                    f"support almost disjoint regions (agreement "
                    f"{agreement:.3f}). Consider a forged or inherited GPS "
                    f"tag, a photograph of another image, or a misattributed "
                    f"file before trusting the ranked candidates."
                )
    return warnings

EARTH_LAND_KM2 = 148_940_000.0


def credible_region_km2(post: np.ndarray, grid: WorldGrid,
                        mass: float = 0.90) -> float:
    """Area of the smallest region holding `mass` of the posterior.

    Entropy in bits is the right internal measure but a poor thing to show an
    analyst: 15.1 against a 17.8-bit uniform world sounds constrained and is
    not. Square kilometres are directly interpretable -- "90% of the mass is
    spread over 5 million km2" is unmistakably a country-level answer, and no
    one mistakes it for a location.
    """
    flat = np.sort(post.ravel())[::-1]
    cum = np.cumsum(flat)
    cutoff = np.searchsorted(cum, mass) + 1
    threshold = flat[min(cutoff, flat.size) - 1]

    selected = post >= threshold
    # Cell area varies with latitude; cell_area is already cos-weighted and
    # normalised, so scale it back to real area.
    lat_km = grid.step * 110.574
    lon_km = grid.step * 111.320 * np.cos(np.radians(grid.lat2d))
    return float((lat_km * lon_km)[selected].sum())


def describe_precision(area_km2: float) -> tuple[str, str]:
    """Map a credible-region area to a plain-language precision band.

    Returns (band, sentence). The bands are deliberately coarse: the point is
    to stop an analyst reading a ranked list as an answer when the evidence
    only supports a continent.
    """
    if area_km2 < 25:
        return ("pinpoint", "Constrained to a specific site.")
    if area_km2 < 2_500:
        return ("locality",
                "Constrained to roughly a town or suburb. The ranked "
                "candidates are meaningful leads.")
    if area_km2 < 100_000:
        return ("regional",
                "Constrained to a region, not a place. Treat candidates as "
                "areas to search, not as locations.")
    if area_km2 < 3_000_000:
        return ("country",
                "Country-level only. The ranked candidates are the most "
                "populated points inside a large area, NOT evidence-backed "
                "locations -- do not read them as leads.")
    return ("unconstrained",
            "Essentially unconstrained. The evidence in this image does not "
            "establish where it was taken; the candidate list is noise.")

