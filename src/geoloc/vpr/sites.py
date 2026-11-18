"""Choosing which sites a corpus covers.

Shared by the command line and the workbench so both produce identical site
identifiers. Identifiers are positional within the cached gazetteer, so two
code paths resolving sites differently would silently file one station under
two names and split its reference images between them.
"""
from __future__ import annotations

from typing import Any


def resolve(country: str, facility: str, near: tuple[float, float] | None = None,
            radius_km: float = 60.0, limit: int = 40) -> list[dict[str, Any]]:
    """Mapped sites of one facility type in a country, nearest first if `near`.

    Raises ValueError for an unknown facility, and OfflineError when the
    gazetteer is not cached and the network is disallowed.
    """
    from ..geo import facility as facility_mod
    from ..geo.grid import haversine_km

    fac = next((f for f in facility_mod.FACILITIES if f.key == facility), None)
    if fac is None:
        raise ValueError(f"unknown facility {facility!r}")

    iso2 = country.upper()
    raw = facility_mod.fetch_sites(iso2, fac)
    sites = [{"id": f"{iso2}_{facility}_{i}",
              "name": s.get("name") or f"site {i}",
              "lat": s["lat"], "lon": s["lon"]}
             for i, s in enumerate(raw)]

    if near is not None:
        lat, lon = near
        sites = [s for s in sites
                 if haversine_km(lat, lon, s["lat"], s["lon"]) <= radius_km]
        sites.sort(key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))

    # Unnamed map features are usually fragments of a named site mapped twice;
    # prefer named ones when any exist.
    named = [s for s in sites if s["name"] and not s["name"].startswith("site ")]
    return (named or sites)[:limit]
