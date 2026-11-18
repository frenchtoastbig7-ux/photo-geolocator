"""Corpus builds that run behind the workbench.

A build harvests reference photographs, embeds them and calibrates the
result: minutes for a small area, far longer for a large one, and mostly
spent waiting on Wikimedia Commons' rate limits. It runs on a background
thread with its progress exposed for polling, and only one runs at a time --
two harvests would share one rate limit and each crawl at half speed, and two
writers could interleave the same manifest.

Cancelling stops the harvest before the next site and keeps every image
already fetched, because the manifest is saved after each site.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from ..config import SETTINGS
from . import calibration, corpus
from . import index as index_mod

_COUNTRY = re.compile(r"[A-Za-z]{2}")


@dataclass
class BuildParams:
    area: str
    country: str
    facility: str
    near: tuple[float, float] | None = None
    radius_km: float = 60.0
    limit: int = 20
    per_site: int = 60
    site_radius_m: int = 450
    use_categories: bool = False


@dataclass
class JobState:
    state: str = "idle"        # idle | running | done | error | cancelled
    phase: str = ""            # resolving | harvesting | embedding | calibrating
    area: str = ""
    message: str = ""
    done: int = 0
    total: int = 0
    images: int = 0
    error: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None


_LOCK = threading.Lock()
_STATE = JobState()
_CANCEL = threading.Event()
_THREAD: threading.Thread | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _update(**changes: Any) -> None:
    with _LOCK:
        for key, value in changes.items():
            setattr(_STATE, key, value)


def _finish(state: str, message: str, *, result: dict[str, Any] | None = None,
            error: str = "") -> None:
    _update(state=state, phase="", message=message, result=result, error=error,
            finished_at=_now())


def status() -> dict[str, Any]:
    with _LOCK:
        return asdict(_STATE)


def parse_params(raw: dict[str, Any]) -> BuildParams:
    """Validate a build request, with messages fit to show an operator."""
    from ..geo.facility import FACILITIES

    area = corpus.validate_area_name(str(raw.get("area", "")).strip())

    country = str(raw.get("country", "")).strip().upper()
    if not _COUNTRY.fullmatch(country):
        raise ValueError("Country must be a two-letter ISO code, such as FR.")

    facility = str(raw.get("facility", "")).strip()
    keys = sorted(f.key for f in FACILITIES)
    if facility not in keys:
        raise ValueError(f"Choose a facility type: {', '.join(keys)}.")

    lat, lon = raw.get("near_lat"), raw.get("near_lon")
    near = None
    if (lat is None) != (lon is None):
        raise ValueError("Give both latitude and longitude for 'near', or neither.")
    if lat is not None:
        try:
            near = (float(lat), float(lon))
        except (TypeError, ValueError):
            raise ValueError("'near' must be numeric latitude and longitude.") from None
        if not (-90.0 <= near[0] <= 90.0 and -180.0 <= near[1] <= 180.0):
            raise ValueError("'near' is outside the valid latitude/longitude range.")

    def number(key: str, cast, low, high, default):
        try:
            value = cast(raw.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number.") from None
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        return value

    return BuildParams(
        area=area, country=country, facility=facility, near=near,
        radius_km=number("radius_km", float, 1, 2000, 60.0),
        limit=number("limit", int, 1, 200, 20),
        per_site=number("per_site", int, 5, 300, 60),
        site_radius_m=number("site_radius_m", int, 50, 3000, 450),
        use_categories=bool(raw.get("use_categories", False)),
    )


def start_build(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and start a build; returns ok, or a reason and an HTTP code."""
    global _THREAD

    if SETTINGS.offline:
        return {"ok": False, "code": 409,
                "reason": ("Offline mode is on. A build downloads reference "
                           "photographs from Wikimedia Commons; turn offline "
                           "mode off to build, then back on.")}
    try:
        params = parse_params(raw)
    except ValueError as exc:
        return {"ok": False, "code": 400, "reason": str(exc)}

    with _LOCK:
        if _STATE.state == "running":
            return {"ok": False, "code": 409,
                    "reason": (f"A build for '{_STATE.area}' is already running. "
                               "Cancel it or wait for it to finish.")}
        _CANCEL.clear()
        fresh = JobState(state="running", phase="resolving", area=params.area,
                         message="Finding mapped sites.", started_at=_now())
        for key, value in asdict(fresh).items():
            setattr(_STATE, key, value)
        _THREAD = threading.Thread(target=_run, args=(params,), daemon=True,
                                   name="corpus-build")
        _THREAD.start()
    return {"ok": True, **status()}


def cancel() -> dict[str, Any]:
    with _LOCK:
        running = _STATE.state == "running"
        if running:
            _CANCEL.set()
            _STATE.message = "Cancelling once the current site finishes."
    return {"ok": running, **status()}


def _run(params: BuildParams) -> None:
    from . import model as vpr_model
    from . import sites as sites_mod

    try:
        found = sites_mod.resolve(params.country, params.facility, params.near,
                                  params.radius_km, params.limit)
        if not found:
            raise ValueError("No mapped sites matched. Widen the radius, drop "
                             "'near', or choose another facility type.")
        _update(phase="harvesting", done=0, total=len(found),
                message=f"Harvesting reference photographs for {len(found)} site(s).")

        def on_site(done: int, total: int, stats: corpus.HarvestStats) -> None:
            _update(done=done, total=total, images=stats.downloaded + stats.skipped,
                    message=f"Harvested {done} of {total} sites.")

        records, stats = corpus.harvest(
            params.area, found, per_site=params.per_site,
            radius_m=params.site_radius_m, use_categories=params.use_categories,
            on_progress=on_site, cancel=_CANCEL)

        if _CANCEL.is_set():
            _finish("cancelled",
                    f"Stopped during the harvest. {len(records)} image(s) are kept "
                    "on disk, but the index was not rebuilt, so matching still "
                    "uses whatever index existed before.")
            return
        if not records:
            raise RuntimeError("The harvest returned no images. Commons may be "
                               "rate-limiting, or these sites have no geotagged "
                               "photographs nearby.")

        note = ("" if vpr_model.is_installed() else
                " The place-recognition model (~0.9 GB) downloads first, once.")
        _update(phase="embedding", done=0, total=len(records),
                message="Embedding reference photographs." + note)
        built = index_mod.build(params.area,
                                on_progress=lambda d, t: _update(done=d, total=t))

        _update(phase="calibrating", done=0, total=0,
                message=("Calibrating: measuring how often this corpus names a "
                         "place it does not contain."))
        cal = calibration.run(params.area)
        _finish("done", "Built and calibrated.",
                result={"images": len(built), "sites": len(built.sites),
                        "failed_downloads": stats.failed, "calibration": cal})
    except Exception as exc:
        _finish("error", "", error=_describe(exc))


def _describe(exc: Exception) -> str:
    from ..config import OfflineError
    from .model import VPRUnavailable

    if isinstance(exc, OfflineError | VPRUnavailable | ValueError | RuntimeError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def list_corpora() -> list[dict[str, Any]]:
    """Every corpus on disk with its coverage and calibration."""
    job = status()
    out: list[dict[str, Any]] = []
    for name, images in corpus.list_areas():
        manifest = corpus.load_manifest(name)
        indexed = index_mod.indexed_count(name)
        cal = calibration.load(name)
        if cal is not None:
            cal = {**cal, "stale": cal.get("images") != indexed}
        out.append({
            "name": name, "images": images,
            "sites": len({r["site_id"] for r in manifest}),
            "indexed": indexed, "calibration": cal,
            "building": job["state"] == "running" and job["area"] == name,
        })
    return out


def _reset_for_tests() -> None:
    global _THREAD
    with _LOCK:
        for key, value in asdict(JobState()).items():
            setattr(_STATE, key, value)
    _CANCEL.clear()
    _THREAD = None
