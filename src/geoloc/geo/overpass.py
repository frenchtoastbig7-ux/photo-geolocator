"""OpenStreetMap refinement.

The *only* component that opens a socket during analysis, and the reason the
network posture is "local-first, opt-in outbound" rather than simply "local".

What crosses the wire: place names read by OCR, and OSM tag filters chosen by
the analyst. What never crosses the wire: the image, any crop of it, any
embedding of it, or any hash of it. That distinction is the whole point --
an adversary observing this traffic learns that someone is looking for a
petrol station near a railway in Croatia, not what photograph prompted it.

Every call is a no-op that raises `OfflineError` when offline mode is on.
"""
from __future__ import annotations

from typing import Any

import httpx

from ..config import SETTINGS, OfflineError

# Feature descriptors an analyst can toggle, mapped to OSM tag filters.
FEATURE_TAGS: dict[str, list[str]] = {
    "church": ['["building"="church"]', '["amenity"="place_of_worship"]'],
    "mosque": ['["amenity"="place_of_worship"]["religion"="muslim"]'],
    "clock tower": ['["man_made"="tower"]["tower:type"="clock"]', '["amenity"="clock"]'],
    "lighthouse": ['["man_made"="lighthouse"]'],
    "windmill": ['["man_made"="windmill"]'],
    "water tower": ['["man_made"="water_tower"]'],
    "railway station": ['["railway"="station"]'],
    "tram stop": ['["railway"="tram_stop"]'],
    "bridge": ['["man_made"="bridge"]'],
    "stadium": ['["leisure"="stadium"]'],
    "castle": ['["historic"="castle"]'],
    "monument": ['["historic"="monument"]'],
    "power plant": ['["power"="plant"]'],
    "wind turbine": ['["power"="generator"]["generator:source"="wind"]'],
    "airport": ['["aeroway"="aerodrome"]'],
    "port / harbour": ['["harbour"="yes"]', '["amenity"="ferry_terminal"]'],
    "university": ['["amenity"="university"]'],
    "hospital": ['["amenity"="hospital"]'],
    "cemetery": ['["landuse"="cemetery"]'],
    "roundabout": ['["junction"="roundabout"]'],
}


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=SETTINGS.http_timeout,
        headers={"User-Agent": SETTINGS.user_agent},
        follow_redirects=True,
    )


def build_query(bbox: tuple[float, float, float, float],
                features: list[str], limit: int = 200) -> str:
    """Overpass QL for the requested features within a bbox.

    bbox is (lat_min, lon_min, lat_max, lon_max), matching Overpass order.
    """
    selectors: list[str] = []
    for feat in features:
        for tag in FEATURE_TAGS.get(feat, []):
            for kind in ("node", "way", "relation"):
                selectors.append(f"  {kind}{tag}({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});")
    if not selectors:
        raise ValueError("no recognised features requested")
    body = "\n".join(selectors)
    return f"[out:json][timeout:60];\n(\n{body}\n);\nout center {limit};"


# The main public Overpass instance is regularly saturated and answers 504
# or 429 rather than failing outright. Mirrors are tried in order so a busy
# endpoint degrades into a slower answer instead of a dead feature.
OVERPASS_MIRRORS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


def _overpass_endpoints() -> list[str]:
    """Configured endpoint first, then the remaining public mirrors."""
    primary = SETTINGS.overpass_endpoint
    return [primary] + [m for m in OVERPASS_MIRRORS if m != primary]


def find_features(bbox: tuple[float, float, float, float],
                  features: list[str], limit: int = 200) -> list[dict[str, Any]]:
    """Query Overpass for features in a bounding box.

    Falls through the mirror list on the overload responses public instances
    return under load (429/502/503/504) and on transport errors.
    """
    if not SETTINGS.net_allowed():
        raise OfflineError(
            "Overpass lookup blocked: offline mode is enabled. Disable it "
            "explicitly if you accept sending this query to a third party."
        )
    query = build_query(bbox, features, limit)

    data = None
    failures: list[str] = []
    for endpoint in _overpass_endpoints():
        try:
            with _client() as c:
                resp = c.post(endpoint, data={"data": query})
            if resp.status_code in (429, 502, 503, 504):
                failures.append(f"{endpoint}: HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            failures.append(f"{endpoint}: {type(exc).__name__}")
            continue

    if data is None:
        raise RuntimeError(
            "every Overpass endpoint refused the query (public instances are "
            "frequently rate-limited). Tried -- " + "; ".join(failures)
        )

    out = []
    for el in data.get("elements", []):
        centre = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if centre.get("lat") is None:
            continue
        tags = el.get("tags", {})
        out.append({
            "osm_type": el.get("type"), "osm_id": el.get("id"),
            "lat": float(centre["lat"]), "lon": float(centre["lon"]),
            "name": tags.get("name", ""), "tags": tags,
            "url": f"https://www.openstreetmap.org/{el.get('type')}/{el.get('id')}",
        })
    return out


def search_place_name(name: str, *, country_codes: list[str] | None = None,
                      limit: int = 10) -> list[dict[str, Any]]:
    """Geocode a place/street name lifted from image text via Nominatim.

    Typically the highest-yield outbound query in the whole tool: one legible
    street name plus a country shortlist often resolves the case outright.
    """
    if not SETTINGS.net_allowed():
        raise OfflineError("Nominatim lookup blocked: offline mode is enabled.")
    params: dict[str, Any] = {"q": name, "format": "jsonv2", "limit": limit,
                              "addressdetails": 1}
    if country_codes:
        params["countrycodes"] = ",".join(c.lower() for c in country_codes)
    with _client() as c:
        resp = c.get(SETTINGS.nominatim_endpoint, params=params)
        resp.raise_for_status()
        results = resp.json()

    return [{
        "display_name": r.get("display_name", ""),
        "lat": float(r["lat"]), "lon": float(r["lon"]),
        "type": r.get("type", ""), "category": r.get("category", ""),
        "importance": r.get("importance", 0.0),
        "country_code": (r.get("address", {}) or {}).get("country_code", "").upper(),
        "url": f"https://www.openstreetmap.org/?mlat={r['lat']}&mlon={r['lon']}#map=16/{r['lat']}/{r['lon']}",
    } for r in results if r.get("lat")]


def bbox_around(lat: float, lon: float, radius_km: float
                ) -> tuple[float, float, float, float]:
    import math
    dlat = radius_km / 111.0
    dlon = radius_km / max(111.0 * math.cos(math.radians(lat)), 1.0)
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)
