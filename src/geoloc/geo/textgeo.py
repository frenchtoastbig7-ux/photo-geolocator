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
def _facility_keywords() -> set[str]:
    """Every word that names a *kind* of place.

    Such a word is by construction useless for geocoding -- it identifies a
    category, not a location -- so the two lists must never diverge. They did:
    "platform" was a facility keyword but not a generic token, so "PLATFORM 4"
    was geocoded, matched a business in Australia, and dragged an entire case
    from France to Victoria.
    """
    from .facility import FACILITIES

    words: set[str] = set()
    for fac in FACILITIES:
        for kw in fac.keywords:
            words.update(w for w in kw.split() if len(w) > 1)
    return words


GENERIC_TOKENS = {
    "library", "business", "school", "college", "university", "hospital",
    "entrance", "exit", "reception", "parking", "car", "park", "toilets",
    "welcome", "open", "closed", "cafe", "bar", "restaurant", "hotel",
    "pharmacy", "bank", "atm", "police", "station", "airport", "museum",
    "centre", "center", "office", "building", "level", "floor", "street",
    "road", "avenue", "stop", "bus", "train", "danger", "warning", "private",
    "no", "yes", "in", "out", "up", "down", "left", "right", "the", "and",
    "platform", "concourse", "terminal", "gate", "arrivals", "departures",
} | _facility_keywords()

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
    # An unrestricted search is worse than none: "BÄCKEREI SCHMIDT" with no
    # country filter returned a bakery in Brazil, one in Utah and one in
    # Novosibirsk, and the first of them became a location constraint. If the
    # evidence has not settled on any country, the honest move is to skip.
    if not country_codes:
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

def geocode_postcodes(evidence: list[Evidence],
                      country_codes: list[str] | None = None) -> list[Evidence]:
    """Resolve a postcode-shaped token to a town, once the country is known.

    This is the step that makes an ambiguous postcode worth anything. Five
    bare digits fit Germany, France, Spain, Italy, Finland and the United
    States, so the token alone is nearly useless -- but a postcode partitions
    a country into thousands of cells, so pairing it with a country the rest
    of the evidence already supports usually lands within a few kilometres.

    Deliberately restricted to the leading countries rather than searched
    worldwide: "75011" resolves somewhere in almost every country that uses
    five digits, and picking whichever answered first would be arbitrary.
    """
    if not SETTINGS.net_allowed() or not country_codes:
        return []

    postal = next((e for e in evidence if e.id == "ocr.postal"), None)
    if postal is None:
        return []
    by_country: dict[str, list[str]] = postal.raw.get("by_country", {})

    out: list[Evidence] = []
    for i, iso2 in enumerate(country_codes[:2]):
        tokens = by_country.get(iso2) or []
        if not tokens:
            continue
        token = tokens[0]
        if i:
            time.sleep(NOMINATIM_MIN_INTERVAL)
        try:
            results = overpass.search_place_name(
                token, country_codes=[iso2], limit=5)
        except Exception:
            continue
        if not results:
            continue

        best = max(results, key=lambda r: r.get("importance") or 0.0)
        distinct = len({(round(r["lat"], 2), round(r["lon"], 2)) for r in results})
        # A postcode should resolve to one place within a country. Several
        # distinct hits means the token was probably not a postcode at all.
        #
        # Capped at MEDIUM when the code's *shape* did not identify the
        # country, because then the country came from other evidence and this
        # only refines it. Emitting HIGH there inverted the hierarchy: five
        # digits resolved in the wrong country produced a confident point that
        # overrode the plate and road numbering which had chosen it.
        ambiguous = bool(postal.raw.get("ambiguous"))
        if distinct != 1:
            confidence = Confidence.MEDIUM if not ambiguous else Confidence.LOW
        else:
            confidence = Confidence.MEDIUM if ambiguous else Confidence.HIGH

        out.append(Evidence(
            id=f"geo.postal.{iso2}", analyzer="geocode",
            title=f"Postcode {token} resolved in {iso2}",
            detail=(f"{token} in {iso2} resolves to "
                    f"{best.get('display_name', '')} "
                    f"({best['lat']:.4f}, {best['lon']:.4f}). "
                    f"{len(results)} result(s), {distinct} distinct location(s)."
                    + (" A postcode narrows to a few kilometres, so this is "
                       "the strongest lead available short of a GPS tag."
                       if distinct == 1 else
                       " Several distinct matches: the token may not be a "
                       "postcode.")
                    + " Only the code was sent; the image was not."),
            confidence=confidence, tags=["text", "postal", "geocode", "network"],
            raw={"postcode": token, "country": iso2, "results": results[:5]},
            constraints=[GeoConstraint(
                kind=ConstraintKind.POINT, lat=float(best["lat"]),
                lon=float(best["lon"]),
                radius_km=6.0 if distinct == 1 else 20.0,
                confidence=confidence,
                note=f"postcode {token} in {iso2}")],
        ))
    return out

