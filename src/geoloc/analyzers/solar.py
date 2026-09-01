"""Solar geometry ("shadow casting") geolocation.

Given when a photo was taken and the direction/length of a shadow in it, the
sun's position is a hard astronomical constraint on where the camera stood.
This is one of the few OSINT techniques that produces a genuinely rigorous
bound rather than a plausibility judgement.

Two regimes, and the difference matters:

* **UTC instant known** (EXIF timestamp with a UTC offset). Solar azimuth and
  elevation together pin *both* latitude and longitude -- often to a band a
  few tens of kilometres wide. Very strong.
* **Date known, time-of-day unknown.** Scanning over the unknown hour
  collapses the longitude information entirely: azimuth+elevation then yields
  latitude and *local solar time*, not longitude. The result is a
  latitude-only constraint, and it is reported as such rather than being
  dressed up as a location.

Shadow azimuth is measured clockwise from true north, pointing *along the
shadow, away from the object*. Analysts must correct for magnetic
declination and for any image rotation before entering a bearing.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from ..models import Confidence, ConstraintKind, Evidence, GeoConstraint


def to_julian_day(dt: datetime) -> float:
    """Julian Day from a timezone-aware datetime."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    dt = dt.astimezone(UTC)
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    day = (dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24.0)
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + day + b - 1524.5)


def solar_position(jd: float, lat: np.ndarray, lon: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (elevation_deg, azimuth_deg) of the sun.

    Low-precision USNO/NOAA algorithm; accurate to roughly 0.01 deg over
    1950-2050, which is far finer than any shadow bearing an analyst can
    read off a photograph.

    Azimuth is measured clockwise from true north.
    """
    n = jd - 2451545.0

    mean_long = (280.460 + 0.9856474 * n) % 360.0
    mean_anom = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_long = math.radians(
        mean_long + 1.915 * math.sin(mean_anom) + 0.020 * math.sin(2 * mean_anom)
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)

    right_asc = math.atan2(math.cos(obliquity) * math.sin(ecl_long), math.cos(ecl_long))
    declination = math.asin(math.sin(obliquity) * math.sin(ecl_long))

    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lmst_deg = (gmst * 15.0 + lon) % 360.0
    hour_angle = np.radians((lmst_deg - math.degrees(right_asc) + 180.0) % 360.0 - 180.0)

    lat_r = np.radians(lat)
    sin_el = (np.sin(lat_r) * math.sin(declination)
              + np.cos(lat_r) * math.cos(declination) * np.cos(hour_angle))
    elevation = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    azimuth = np.degrees(np.arctan2(
        -np.sin(hour_angle) * math.cos(declination),
        np.cos(lat_r) * math.sin(declination)
        - np.sin(lat_r) * math.cos(declination) * np.cos(hour_angle),
    )) % 360.0
    return elevation, azimuth


def elevation_from_shadow_ratio(object_height: float, shadow_length: float) -> float:
    """Solar elevation from an object height and its shadow length.

    Units cancel, so any consistent measure works -- including pixel counts,
    provided the object is vertical and the ground is level and the shot is
    not badly foreshortened.
    """
    if shadow_length <= 0:
        raise ValueError("shadow length must be positive")
    return math.degrees(math.atan2(object_height, shadow_length))


def _angular_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest signed difference between bearings, in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def shadow_heatmap(grid, *, when_utc: datetime | None = None,
                   date_only: datetime | None = None,
                   shadow_azimuth: float | None = None,
                   sun_elevation: float | None = None,
                   azimuth_sigma: float = 8.0,
                   elevation_sigma: float = 5.0) -> tuple[np.ndarray, str]:
    """Likelihood field from a shadow observation.

    Returns (field, mode) where mode is 'instant' or 'date-only'.
    """
    sun_azimuth = None
    if shadow_azimuth is not None:
        # The shadow points directly away from the sun.
        sun_azimuth = (shadow_azimuth + 180.0) % 360.0

    def score(jd: float) -> np.ndarray:
        el, az = solar_position(jd, grid.lat2d, grid.lon2d)
        s = np.ones(grid.shape)
        if sun_azimuth is not None:
            s *= np.exp(-0.5 * (_angular_diff(az, sun_azimuth) / azimuth_sigma) ** 2)
        if sun_elevation is not None:
            s *= np.exp(-0.5 * ((el - sun_elevation) / elevation_sigma) ** 2)
        # The sun must actually be up to cast a shadow.
        s *= (el > -0.9)
        return s

    if when_utc is not None:
        return score(to_julian_day(when_utc)), "instant"

    if date_only is not None:
        # Marginalise over the unknown time of day. Every longitude ends up
        # equally supported, which is the honest result: without a clock,
        # shadows carry latitude information only.
        base = date_only.replace(hour=0, minute=0, second=0, microsecond=0,
                                 tzinfo=UTC)
        jd0 = to_julian_day(base)
        acc = np.zeros(grid.shape)
        for step in range(0, 24 * 6):  # 10-minute resolution
            acc = np.maximum(acc, score(jd0 + step / (24.0 * 6.0)))
        # Collapse to latitude-only to avoid implying spurious longitude
        # structure from the discrete time scan.
        lat_profile = acc.max(axis=1, keepdims=True)
        return np.repeat(lat_profile, grid.shape[1], axis=1), "date-only"

    return np.ones(grid.shape), "unconstrained"


def analyze(grid, *, when_utc: datetime | None = None,
            date_only: datetime | None = None,
            shadow_azimuth: float | None = None,
            sun_elevation: float | None = None,
            object_height: float | None = None,
            shadow_length: float | None = None,
            heatmaps: dict[str, np.ndarray] | None = None) -> list[Evidence]:
    """Build solar evidence from analyst-supplied shadow measurements.

    Shadow measurement is left to the analyst rather than automated: reliable
    automatic extraction of a shadow bearing from a single uncalibrated
    photograph is not a solved problem, and a wrong bearing here produces a
    confidently wrong answer several hundred kilometres out.
    """
    if sun_elevation is None and object_height and shadow_length:
        sun_elevation = elevation_from_shadow_ratio(object_height, shadow_length)

    if shadow_azimuth is None and sun_elevation is None:
        return []
    if when_utc is None and date_only is None:
        return [Evidence(
            id="solar.no_time", analyzer="solar",
            title="Shadow measured but no date available",
            detail="A shadow bearing without a date cannot constrain location: "
                   "the same shadow occurs at different latitudes on different "
                   "days. Supply at least a capture date.",
            confidence=Confidence.LOW, tags=["solar"],
        )]

    field, mode = shadow_heatmap(
        grid, when_utc=when_utc, date_only=date_only,
        shadow_azimuth=shadow_azimuth, sun_elevation=sun_elevation,
    )
    if heatmaps is not None:
        heatmaps["solar"] = field

    bits = []
    if shadow_azimuth is not None:
        bits.append(f"shadow bearing {shadow_azimuth:.0f} deg "
                    f"(sun at {(shadow_azimuth + 180) % 360:.0f} deg)")
    if sun_elevation is not None:
        bits.append(f"sun elevation {sun_elevation:.1f} deg")
    if object_height and shadow_length:
        bits.append(f"from height:shadow ratio {object_height:g}:{shadow_length:g}")

    if mode == "instant":
        detail = (f"Solved for a known UTC instant "
                  f"({when_utc.isoformat()}): {', '.join(bits)}. "
                  "Azimuth and elevation together constrain both latitude "
                  "and longitude.")
        conf = Confidence.HIGH
    else:
        detail = (f"Date known but time-of-day unknown "
                  f"({date_only.date().isoformat()}): {', '.join(bits)}. "
                  "Marginalising over the unknown hour leaves a "
                  "LATITUDE-ONLY constraint -- longitude is not determined.")
        conf = Confidence.MEDIUM

    return [Evidence(
        id="solar.shadow", analyzer="solar",
        title=f"Solar geometry ({mode})",
        detail=detail, confidence=conf, tags=["solar", "geometry"],
        raw={"mode": mode, "shadow_azimuth": shadow_azimuth,
             "sun_elevation": sun_elevation,
             "when_utc": when_utc.isoformat() if when_utc else None,
             "date_only": date_only.date().isoformat() if date_only else None},
        constraints=[GeoConstraint(
            kind=ConstraintKind.HEATMAP, heatmap_ref="solar",
            confidence=conf, note="solar position likelihood",
        )],
    )]


def local_solar_time(lat: float, lon: float, date: datetime,
                     shadow_azimuth: float,
                     sun_elevation: float | None = None,
                     step_minutes: int = 5) -> tuple[datetime, float, float] | None:
    """When, on `date`, would a shadow at this place point that way?

    Runs the solar solver forwards over the day at a candidate location and
    returns the instant whose predicted shadow best matches the observed
    bearing. This is the inverse of the geolocation use: once a location is
    proposed, the same geometry dates the photograph.

    Returns (utc_instant, solar_elevation_deg, bearing_error_deg), or None if
    the sun is never in the right part of the sky that day -- which is itself
    informative, since it rules the candidate out.
    """
    sun_azimuth = (shadow_azimuth + 180.0) % 360.0
    base = date.replace(hour=0, minute=0, second=0, microsecond=0,
                        tzinfo=UTC)
    jd0 = to_julian_day(base)

    lat_a = np.array([lat])
    lon_a = np.array([lon])
    best: tuple[datetime, float, float] | None = None

    steps = int(24 * 60 / step_minutes)
    for i in range(steps):
        jd = jd0 + (i * step_minutes) / (24.0 * 60.0)
        el, az = solar_position(jd, lat_a, lon_a)
        elevation = float(el[0])
        if elevation <= 0.5:          # sun down: no shadow to cast
            continue
        err = abs(float(_angular_diff(az, sun_azimuth)[0]))
        if sun_elevation is not None:
            err += abs(elevation - sun_elevation)
        if best is None or err < best[2]:
            from datetime import timedelta
            best = (base + timedelta(minutes=i * step_minutes), elevation, err)
    return best


def solar_noon_utc(lat: float, lon: float, date: datetime) -> datetime | None:
    """UTC instant of local solar noon, for converting to local solar time."""
    base = date.replace(hour=0, minute=0, second=0, microsecond=0,
                        tzinfo=UTC)
    jd0 = to_julian_day(base)
    lat_a, lon_a = np.array([lat]), np.array([lon])
    best = None
    for i in range(24 * 12):          # 5-minute resolution
        jd = jd0 + (i * 5) / (24.0 * 60.0)
        el, _ = solar_position(jd, lat_a, lon_a)
        if best is None or float(el[0]) > best[1]:
            from datetime import timedelta
            best = (base + timedelta(minutes=i * 5), float(el[0]))
    return best[0] if best else None

def times_for_elevation(lat: float, lon: float, date: datetime,
                        elevation_deg: float, step_minutes: int = 2
                        ) -> tuple[datetime, datetime] | None:
    """The two instants on `date` when the sun sits at a given height.

    Elevation is the one solar quantity recoverable from a photograph without
    knowing which way the camera faced, which makes it the only route to a
    capture time for an image whose metadata has been stripped. The cost is
    an irreducible ambiguity: the sun passes every height twice, once
    climbing and once descending, so this returns both and leaves the choice
    to other evidence.

    Returns None when the sun never reaches that height on that date at that
    latitude -- itself a useful finding, since it means the stated date and
    the observed shadows cannot both be true.
    """
    day = date.replace(hour=0, minute=0, second=0, microsecond=0,
                       tzinfo=date.tzinfo or UTC)
    lat_a, lon_a = np.array([lat]), np.array([lon])

    samples: list[tuple[datetime, float]] = []
    for minute in range(0, 24 * 60, step_minutes):
        when = day + timedelta(minutes=minute)
        elev, _az = solar_position(to_julian_day(when), lat_a, lon_a)
        samples.append((when, float(elev[0])))

    peak = max(s[1] for s in samples)
    if peak < elevation_deg:
        return None

    noon_index = max(range(len(samples)), key=lambda i: samples[i][1])
    morning = min(samples[:noon_index + 1],
                  key=lambda s: abs(s[1] - elevation_deg), default=None)
    afternoon = min(samples[noon_index:],
                    key=lambda s: abs(s[1] - elevation_deg), default=None)
    if morning is None or afternoon is None:
        return None
    return morning[0], afternoon[0]


def max_elevation_on(lat: float, lon: float, date: datetime) -> float:
    """Highest the sun gets at this place on this date."""
    day = date.replace(hour=0, minute=0, second=0, microsecond=0,
                       tzinfo=date.tzinfo or UTC)
    lat_a, lon_a = np.array([lat]), np.array([lon])
    best = -90.0
    for minute in range(0, 24 * 60, 5):
        elev, _ = solar_position(to_julian_day(day + timedelta(minutes=minute)),
                                 lat_a, lon_a)
        best = max(best, float(elev[0]))
    return best


def date_window_for_elevation(lat: float, lon: float, elevation_deg: float,
                              year: int, step_days: int = 3
                              ) -> tuple[datetime, datetime] | None:
    """Dates on which the sun can reach a given height at a given latitude.

    A measured sun elevation dates a photograph as well as timing it. The
    sun's maximum height varies through the year by twice the axial tilt, so
    an elevation near the local summer maximum is only attainable for part of
    the year -- at 43°N, 47° is unreachable at the equinox and impossible in
    winter, which places the photograph between roughly April and August
    without any metadata at all.

    Returns the first and last date the height is attainable, or None if it
    is attainable all year (tropics) or never.
    """
    attainable = []
    for day_of_year in range(1, 366, step_days):
        date = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1)
        if max_elevation_on(lat, lon, date) >= elevation_deg:
            attainable.append(date)
    if not attainable or len(attainable) >= (365 // step_days):
        return None
    return attainable[0], attainable[-1]
