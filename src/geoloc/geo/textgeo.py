"""Turn text read out of the image into a location constraint.

The workbench could already geocode a name the analyst typed, but nothing
fed that back into the posterior -- so a photograph with a legible shop name
on it produced no better an answer than a blank wall. This closes that loop
automatically, which matters because legible local text is the most reliable
non-metadata cue there is: far more dependable than any scene model.

Only derived text leaves the machine, never image data, and only when
outbound access is permitted.
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..config import SETTINGS
from ..models import Confidence, ConstraintKind, Evidence, GeoConstraint
from . import overpass

# Words that appear on buildings everywhere and geocode to arbitrary places.
# Querying "LIBRARY" returns a library; it just will not be the right one.
GENERIC_TOKENS = {
    "library", "business", "school", "college", "university", "hospital",
    "entrance", "exit", "reception", "parking", "car", "park", "toilets",
    "welcome", "open", "closed", "cafe", "bar", "restaurant", "hotel",
    "pharmacy", "bank", "atm", "police", "station", "airport", "museum",
    "centre", "center", "office", "building", "level", "floor", "street",
    "road", "avenue", "stop", "bus", "train", "danger", "warning", "private",
    "no", "yes", "in", "out", "up", "down", "left", "right", "the", "and",
}

MIN_CHARS = 5
MAX_QUERIES = 4
NOMINATIM_MIN_INTERVAL = 1.1  # the public instance asks for <= 1 req/sec


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\n\r-\u2013\u2014\u00b7|,.")


def _informative(text: str) -> bool:
    """Whether a string is specific enough to be worth geocoding."""
    t = _clean(text)
    if len(t) < MIN_CHARS:
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 4:
        return False
    words = [w for w in re.split(r"[^\w']+", t.lower()) if w]
    if not words:
        return False
    # Every word generic -> the query cannot identify anywhere in particular.
    if all(w in GENERIC_TOKENS or w.isdigit() for w in words):
        return False
    # A single short generic-ish word is not worth a lookup either.
    return not (len(words) == 1 and len(words[0]) < 6)


def candidate_queries(evidence: list[Evidence]) -> list[str]:
    """Pick the most promising strings from OCR output, best first."""
    blocks: list[tuple[str, float]] = []
    for ev in evidence:
        if ev.analyzer != "text" or ev.id != "ocr.text":
            continue
        for b in ev.raw.get("blocks", []):
            text = _clean(str(b.get("text", "")))
            if _informative(text):
                blocks.append((text, float(b.get("confidence", 0.0))))

    blocks.sort(key=lambda kv: (-kv[1], -len(kv[0])))
    seen: set[str] = set()
    out: list[str] = []
    for text, _conf in blocks:
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)

    # A shopfront often splits across lines ("PEKARNA" / "DUBRAVICA"); the
    # joined phrase is usually the more identifying query.
    if len(out) >= 2:
        joined = _clean(f"{out[0]} {out[1]}")
        if joined.lower() not in seen and len(joined) <= 80:
            out.insert(0, joined)
    return out[:MAX_QUERIES]


def geocode_evidence(evidence: list[Evidence], country_codes: list[str] | None = None,
                     *, max_queries: int = MAX_QUERIES) -> list[Evidence]:
    """Geocode promising OCR strings and return them as located evidence."""
    if not SETTINGS.net_allowed():
        return []

    queries = candidate_queries(evidence)[:max_queries]
    if not queries:
        return []

    out: list[Evidence] = []
    for i, query in enumerate(queries):
        if i:
            time.sleep(NOMINATIM_MIN_INTERVAL)
        try:
            results: list[dict[str, Any]] = overpass.search_place_name(
                query, country_codes=country_codes, limit=5)
        except Exception:
            continue
        if not results:
            continue

        best = max(results, key=lambda r: r.get("importance") or 0.0)
        # Several equally-plausible hits mean the name is not distinctive
        # enough to locate anything; a single strong hit is the useful case.
        distinct = len({(round(r["lat"], 1), round(r["lon"], 1)) for r in results})
        confidence = Confidence.MEDIUM if distinct == 1 else Confidence.LOW

        out.append(Evidence(
            id=f"geo.text.{i}", analyzer="geocode",
            title=f"Text geocoded: {query!r}",
            detail=(f"Nominatim resolved {query!r} to "
                    f"{best.get('display_name', '')} "
                    f"({best['lat']:.5f}, {best['lon']:.5f}). "
                    f"{len(results)} result(s), {distinct} distinct location(s)"
                    + (". A single match makes this a strong lead."
                       if distinct == 1 else
                       ". Multiple candidate locations -- weak on its own.")
                    + " Only this text was sent; the image was not."),
            confidence=confidence,
            tags=["text", "geocode", "network"],
            raw={"query": query, "results": results[:5]},
            constraints=[GeoConstraint(
                kind=ConstraintKind.POINT, lat=float(best["lat"]),
                lon=float(best["lon"]), radius_km=3.0,
                confidence=confidence, note=f"geocoded text: {query!r}",
            )],
        ))
    return out
