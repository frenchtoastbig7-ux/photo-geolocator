"""Local web workbench.

Binds to loopback by default. There is no authentication because there is no
intended remote user: this is a single-analyst desktop tool, and exposing it
on a routable interface would be a mistake regardless of any auth layer.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import modelmgr
from .config import SETTINGS, OfflineError
from .geo import overpass
from .pipeline import AnalystInput, analyze_image, load_case
from .report import write_report
from .viz import posterior_cells

WEB = Path(__file__).parent / "web"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".bmp"}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024

class RevalidatingStatic(StaticFiles):
    """Static files that must be revalidated on every request.

    The workbench is developed and tweaked in place, and a browser holding a
    stale `app.js` produces confusing, hard-to-diagnose behaviour. ETags make
    revalidation cheap, and this is loopback traffic regardless.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app = FastAPI(title="geoloc workbench", docs_url=None, redoc_url=None)
app.mount("/static", RevalidatingStatic(directory=WEB / "static"), name="static")


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return {
        "offline": SETTINGS.offline,
        "device": SETTINGS.resolve_device(),
        "clip_model": SETTINGS.clip_model,
        "grid_step": SETTINGS.grid_step,
        "blur_faces": SETTINGS.blur_faces_in_exports,
        "case_dir": str(SETTINGS.case_dir),
        "features": sorted(overpass.FEATURE_TAGS),
    }


@app.post("/api/settings")
async def set_settings(request: Request) -> dict[str, Any]:
    body = await request.json()
    if "offline" in body:
        SETTINGS.offline = bool(body["offline"])
    if "blur_faces" in body:
        SETTINGS.blur_faces_in_exports = bool(body["blur_faces"])
    return get_settings()


@app.get("/api/model")
def api_model_status() -> dict[str, Any]:
    """Whether the optional scene model is installed, and download progress."""
    return modelmgr.status()


@app.post("/api/model/install")
def api_model_install() -> dict[str, Any]:
    result = modelmgr.start_download()
    if not result.get("ok"):
        raise HTTPException(409, result.get("reason", "cannot start download"))
    return {**result, **modelmgr.status()}


@app.delete("/api/model")
def api_model_remove() -> dict[str, Any]:
    return {**modelmgr.remove(), **modelmgr.status()}


@app.get("/api/corpus")
def api_corpus() -> dict[str, Any]:
    """Reference corpora on disk, any build in progress, and build options."""
    from .geo.facility import FACILITIES
    from .vpr import calibration, jobs
    from .vpr import model as vpr_model

    return {
        "areas": jobs.list_corpora(),
        "job": jobs.status(),
        "vpr_model_installed": vpr_model.is_installed(),
        "offline": SETTINGS.offline,
        "facilities": [{"key": f.key, "label": f.label} for f in FACILITIES],
        "acceptable_fabrication": calibration.ACCEPTABLE_FABRICATION,
    }


@app.get("/api/corpus/job")
def api_corpus_job() -> dict[str, Any]:
    """Progress of the current or most recent build. Cheap enough to poll."""
    from .vpr import jobs

    return jobs.status()


@app.post("/api/corpus/build")
async def api_corpus_build(request: Request) -> dict[str, Any]:
    from .vpr import jobs

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Expected a JSON body.") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object.")
    result = jobs.start_build(body)
    if not result.get("ok"):
        raise HTTPException(result.get("code", 400),
                            result.get("reason", "Cannot start the build."))
    return result


@app.post("/api/corpus/cancel")
def api_corpus_cancel() -> dict[str, Any]:
    from .vpr import jobs

    return jobs.cancel()


@app.post("/api/corpus/{area}/calibrate")
def api_corpus_calibrate(area: str) -> dict[str, Any]:
    from .vpr import calibration, corpus, jobs

    try:
        corpus.validate_area_name(area)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    job = jobs.status()
    if job["state"] == "running" and job["area"] == area:
        raise HTTPException(409, "This corpus is still being built; it is "
                                 "calibrated automatically when the build ends.")
    try:
        return calibration.run(area)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None


@app.delete("/api/corpus/{area}")
def api_corpus_delete(area: str) -> dict[str, Any]:
    from .vpr import corpus, jobs

    try:
        corpus.validate_area_name(area)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    job = jobs.status()
    if job["state"] == "running" and job["area"] == area:
        raise HTTPException(409, "Cancel the build before deleting this corpus.")
    try:
        removed = corpus.delete_area(area)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    return {"ok": True, "area": area, "deleted_files": removed}


@app.get("/api/cases")
def list_cases() -> list[dict[str, Any]]:
    SETTINGS.ensure_dirs()
    out = []
    for d in sorted(SETTINGS.case_dir.iterdir(), reverse=True):
        meta = d / "case.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        top = (data.get("candidates") or [{}])[0]
        out.append({
            "case_id": data.get("case_id", d.name),
            "created_utc": data.get("created_utc"),
            "filename": (data.get("image") or {}).get("filename", ""),
            "top": top.get("label", ""),
            "entropy_bits": data.get("entropy_bits"),
        })
    return out[:200]


@app.post("/api/analyze")
async def api_analyze(
    image: UploadFile = File(...),
    driving_side: str = Form(""),
    shadow_azimuth: str = Form(""),
    sun_elevation: str = Form(""),
    object_height: str = Form(""),
    shadow_length: str = Form(""),
    capture_utc: str = Form(""),
    capture_date: str = Form(""),
    countries: str = Form(""),
    exclude: str = Form(""),
    notes: str = Form(""),
    run_scene: str = Form("true"),
    habitation_prior: str = Form("true"),
) -> JSONResponse:
    suffix = Path(image.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported image type: {suffix or 'none'}")

    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB")

    tmp_dir = Path(tempfile.mkdtemp(prefix="geoloc-upload-"))
    tmp_path = tmp_dir / Path(image.filename or "upload").name
    try:
        tmp_path.write_bytes(data)
        analyst = AnalystInput(
            driving_side=driving_side if driving_side in {"left", "right"} else None,
            shadow_azimuth=_f(shadow_azimuth), sun_elevation=_f(sun_elevation),
            object_height=_f(object_height), shadow_length=_f(shadow_length),
            capture_datetime_utc=_parse_dt(capture_utc),
            capture_date=_parse_dt(capture_date),
            notes=notes.strip(),
            country_hints=[c.strip().upper() for c in countries.split(",") if c.strip()],
            exclude_countries=[c.strip().upper() for c in exclude.split(",") if c.strip()],
        )
        report = analyze_image(
            tmp_path, analyst=analyst,
            run_scene_model=run_scene.lower() != "false",
            use_habitation_prior=habitation_prior.lower() != "false",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    case_dir = SETTINGS.case_dir / report.case_id
    write_report(report, case_dir)
    post = np.load(case_dir / "posterior.npy")
    from .geo.grid import world_grid

    return JSONResponse({
        "case": json.loads(report.model_dump_json()),
        "cells": posterior_cells(post, world_grid(), top_n=1500),
    })


@app.get("/api/case/{case_id}")
def api_case(case_id: str) -> dict[str, Any]:
    case_dir = _case_dir(case_id)
    report, post = load_case(case_dir)
    from .geo.grid import world_grid

    return {
        "case": json.loads(report.model_dump_json()),
        "cells": posterior_cells(post, world_grid(), top_n=1500),
    }


@app.get("/api/case/{case_id}/file/{name}")
def api_case_file(case_id: str, name: str) -> FileResponse:
    case_dir = _case_dir(case_id)
    # Resolve and confirm containment; `name` comes from the browser.
    target = (case_dir / name).resolve()
    if not str(target).startswith(str(case_dir.resolve()) + "/") or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.post("/api/osm/geocode")
async def api_geocode(request: Request) -> dict[str, Any]:
    body = await request.json()
    name = (body.get("q") or "").strip()
    if not name:
        raise HTTPException(400, "empty query")
    try:
        results = overpass.search_place_name(
            name, country_codes=body.get("countries") or None)
    except OfflineError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"geocoder error: {exc}") from exc
    return {"query": name, "results": results}


@app.post("/api/osm/features")
async def api_features(request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        lat, lon = float(body["lat"]), float(body["lon"])
        radius = float(body.get("radius_km", 15))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "lat, lon required") from exc
    features = [f for f in (body.get("features") or []) if f in overpass.FEATURE_TAGS]
    if not features:
        raise HTTPException(400, "no valid features requested")
    bbox = overpass.bbox_around(lat, lon, radius)
    try:
        results = overpass.find_features(bbox, features)
    except OfflineError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"overpass error: {exc}") from exc
    return {"bbox": bbox, "count": len(results), "results": results}


def _case_dir(case_id: str) -> Path:
    # case_id reaches us from the URL; never let it escape the case root.
    candidate = (SETTINGS.case_dir / case_id).resolve()
    root = SETTINGS.case_dir.resolve()
    if not str(candidate).startswith(str(root) + "/") or not candidate.is_dir():
        raise HTTPException(404, "unknown case")
    return candidate
