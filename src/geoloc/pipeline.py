"""Analysis orchestration.

Runs the analyzers, folds in whatever the analyst has determined by hand,
fuses everything into a posterior, and materialises a case directory.

Analyst input is treated as first-class evidence and generally outranks the
models. A human who has identified the driving side, read a street name, or
measured a shadow bearing has produced better evidence than any zero-shot
classifier in this pipeline, and the weighting reflects that.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import metaverify
from .analyzers import clip_scene, faces, metadata, shadows, solar, text_ocr
from .config import SETTINGS
from .geo import facility as facility_mod
from .geo import fusion, textgeo
from .geo.grid import world_grid
from .models import (
    CaseReport,
    Confidence,
    ConstraintKind,
    Evidence,
    GeoConstraint,
    ImageFacts,
    new_case_id,
    sha256_file,
)


@dataclass
class AnalystInput:
    """Observations the analyst supplies by hand."""

    driving_side: str | None = None          # 'left' | 'right'
    shadow_azimuth: float | None = None      # degrees clockwise from true north
    sun_elevation: float | None = None       # degrees above horizon
    object_height: float | None = None       # any consistent unit
    shadow_length: float | None = None       # same unit as object_height
    capture_datetime_utc: datetime | None = None
    capture_date: datetime | None = None
    notes: str = ""
    facility_type: str | None = None
    """A facility key from geo.facility.FACILITIES. Set when the analyst can
    tell what kind of place this is; it unlocks the site gazetteer, which the
    scene model is not reliable enough to trigger on its own."""
    country_hints: list[str] = field(default_factory=list)
    exclude_countries: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None  # lat_min, lat_max, lon_min, lon_max


def image_facts(path: Path) -> ImageFacts:
    with Image.open(path) as im:
        w, h = im.size
        fmt, mode = im.format or "unknown", im.mode
    return ImageFacts(
        path=str(path.resolve()), filename=path.name,
        sha256=sha256_file(path), bytes=path.stat().st_size,
        width=w, height=h, format=fmt, mode=mode,
    )


def _analyst_evidence(ai: AnalystInput) -> list[Evidence]:
    out: list[Evidence] = []
    if ai.driving_side in {"left", "right"}:
        out.append(clip_scene.driving_side_evidence(ai.driving_side == "left"))

    if ai.country_hints:
        out.append(Evidence(
            id="analyst.country_hint", analyzer="analyst",
            title=f"Analyst country shortlist: {', '.join(ai.country_hints)}",
            detail="Manually supplied candidate countries.",
            confidence=Confidence.HIGH, tags=["analyst"],
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores={c.upper(): 1.0 for c in ai.country_hints},
                floor=0.08, confidence=Confidence.HIGH, note="analyst shortlist",
            )],
        ))

    if ai.exclude_countries:
        out.append(Evidence(
            id="analyst.country_exclude", analyzer="analyst",
            title=f"Analyst exclusions: {', '.join(ai.exclude_countries)}",
            detail="Countries ruled out by the analyst.",
            confidence=Confidence.HIGH, tags=["analyst"],
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores={c.upper(): 0.01 for c in ai.exclude_countries},
                floor=1.0, confidence=Confidence.HIGH, note="analyst exclusion",
            )],
        ))

    if ai.bbox:
        lat_min, lat_max, lon_min, lon_max = ai.bbox
        out.append(Evidence(
            id="analyst.bbox", analyzer="analyst",
            title="Analyst search area",
            detail=f"Restricted to {lat_min:.2f}..{lat_max:.2f} lat, "
                   f"{lon_min:.2f}..{lon_max:.2f} lon.",
            confidence=Confidence.HIGH, tags=["analyst"],
            constraints=[GeoConstraint(
                kind=ConstraintKind.BBOX, lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max, softness_deg=1.0, floor=0.02,
                confidence=Confidence.HIGH, note="analyst search area",
            )],
        ))

    if ai.notes:
        out.append(Evidence(
            id="analyst.notes", analyzer="analyst",
            title="Analyst notes", detail=ai.notes,
            confidence=Confidence.MEDIUM, tags=["analyst"],
        ))
    return out


def _exif_datetime(evidence: list[Evidence]) -> tuple[datetime | None, datetime | None]:
    """Recover a capture instant from metadata evidence, if one survived."""
    ev = next((e for e in evidence if e.id == "meta.timestamp"), None)
    if not ev:
        return None, None
    raw = ev.raw.get("datetime_original", "")
    offset = ev.raw.get("offset", "")
    try:
        naive = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None, None
    if offset:
        try:
            sign = -1 if str(offset).startswith("-") else 1
            body = str(offset).lstrip("+-").split(":")
            delta_h = int(body[0]) + (int(body[1]) / 60 if len(body) > 1 else 0)
            from datetime import timedelta
            return (naive - timedelta(hours=sign * delta_h)).replace(
                tzinfo=UTC), naive.replace(tzinfo=UTC)
        except (ValueError, IndexError):
            pass
    return None, naive.replace(tzinfo=UTC)


def _ocr_texts(report: CaseReport) -> list[str]:
    out: list[str] = []
    for ev in report.evidence:
        if ev.analyzer == "text" and ev.id == "ocr.text":
            out.extend(str(b.get("text", "")) for b in ev.raw.get("blocks", []))
    return out


def _facility_evidence(report: CaseReport, analyst: AnalystInput) -> list[Evidence]:
    """Narrow to known sites of an inferred or analyst-supplied facility type.

    The facility may be asserted by the analyst, or inferred from words OCR
    found. It is deliberately NOT inferred from the scene model: measured on
    this build, CLIP ranks "shopping centre" above "university campus" on a
    university campus, and gives a blank grey image a 37-point margin on
    "industrial estate". Triggering a gazetteer query on that would narrow
    confidently to the wrong kind of place.
    """
    if not SETTINGS.net_allowed():
        return []

    countries = [cc for cc, share in report.top_countries[:2] if share > 0.15]
    if not countries:
        return []

    fac = None
    why = ""
    if analyst.facility_type:
        fac = next((f for f in facility_mod.FACILITIES
                    if f.key == analyst.facility_type), None)
        why = "specified by the analyst"
    if fac is None:
        inferred = facility_mod.infer_facilities(_ocr_texts(report))
        if inferred:
            fac, matched = inferred[0]
            why = f"inferred from text: {', '.join(matched)}"
    if fac is None:
        return []

    out: list[Evidence] = []
    for iso2 in countries:
        try:
            sites = facility_mod.fetch_sites(iso2, fac)
        except Exception:
            continue
        if not sites:
            continue
        coords = [(s["lat"], s["lon"]) for s in sites]
        named = [s for s in sites if s.get("name")]
        out.append(Evidence(
            id=f"facility.{fac.key}.{iso2}", analyzer="facility",
            title=f"{len(sites)} candidate {fac.label} site(s) in {iso2}",
            detail=(f"Facility type {why}. Every {fac.label} recorded in "
                    f"OpenStreetMap for {iso2} was retrieved: {len(sites)} "
                    f"site(s), {len(named)} named. The photograph was almost "
                    "certainly taken at one of these, so the posterior is "
                    "restricted to them. This narrows the search to a list "
                    "you can check against imagery -- it does not pick one."),
            confidence=Confidence.MEDIUM,
            tags=["facility", "gazetteer", "network"],
            raw={"facility": fac.key, "country": iso2,
                 "site_count": len(sites), "sites": sites[:1200]},
            constraints=[GeoConstraint(
                kind=ConstraintKind.SITES, sites=coords,
                site_radius_km=max(fac.typical_extent_km, 1.0),
                floor=1e-4, confidence=Confidence.MEDIUM,
                note=f"{len(sites)} {fac.label} sites in {iso2}",
            )],
        ))
    return out


def _vpr_evidence(image_path: Path, report: CaseReport,
                  area: str | None = None) -> list[Evidence]:
    """Match the image against harvested reference imagery.

    This is the only component that can name a specific place, and it earns
    that by retrieval against real photographs rather than by asking a model
    to pick a label. It abstains by design: a corpus always has a nearest
    neighbour, and returning it unconditionally is precisely how automated
    geolocation produces confident, precise, wrong answers.
    """
    from .vpr import corpus as vpr_corpus
    from .vpr import index as vpr_index
    from .vpr import match as vpr_match
    from .vpr import model as vpr_model

    areas = [area] if area else [name for name, count in vpr_corpus.list_areas() if count]
    if not areas:
        return []
    if not vpr_model.is_installed() and not SETTINGS.net_allowed(purpose="model_download"):
        return []

    try:
        descriptor = vpr_model.embed_image(image_path)
    except vpr_model.VPRUnavailable as exc:
        return [Evidence(
            id="vpr.unavailable", analyzer="vpr",
            title="Place-recognition model not installed", detail=str(exc),
            confidence=Confidence.LOW, tags=["tooling"])]

    out: list[Evidence] = []
    # Corpora may overlap: two areas harvested around the same city both
    # contain its station. Emitting a match from each would count one piece
    # of evidence twice and inflate the posterior, so accepted matches are
    # deduplicated by location and only the strongest is kept.
    accepted: dict[tuple[int, int], tuple[float, Evidence]] = {}

    for name in areas:
        idx = vpr_index.load(name)
        if idx is None or not len(idx):
            continue
        result = vpr_match.match(idx, descriptor)
        best = result.matches[0] if result.matches else None

        if not result.accepted:
            # Reported rather than dropped: "this place is not in the corpus"
            # tells the analyst to widen coverage, and stops a silent
            # near-miss being mistaken for the model having no opinion.
            out.append(Evidence(
                id=f"vpr.nomatch.{name}", analyzer="vpr",
                title=f"No confident visual match in corpus '{name}'",
                detail=result.reason + (
                    f" Nearest was {best.site_name or best.site_id} at "
                    f"{best.best_score:.3f}." if best else ""),
                confidence=Confidence.LOW, tags=["vpr", "corpus"],
                raw={"area": name, **result.as_dict()}))
            continue

        key = (round(best.lat * 200), round(best.lon * 200))   # ~500 m cells
        prior = accepted.get(key)
        if prior is not None and prior[0] >= best.best_score:
            continue
        accepted[key] = (best.best_score, Evidence(
            id=f"vpr.match.{name}", analyzer="vpr",
            title=f"Visual match: {best.site_name or best.site_id}",
            detail=(result.reason + " Matched by image retrieval against "
                    f"harvested reference photographs in corpus '{name}', not "
                    "by a model naming the place. Confirm against independent "
                    "imagery before relying on it."),
            confidence=Confidence.HIGH, tags=["vpr", "corpus"],
            # `sites` is what candidate extraction reads to label a cell by
            # the place found rather than by whatever settlement is nearest,
            # so a match reports "Marseille Saint-Charles" and not "Marseille 03".
            raw={"area": name,
                 "sites": [{"name": best.site_name or best.site_id,
                            "lat": best.lat, "lon": best.lon}],
                 **result.as_dict()},
            constraints=[GeoConstraint(
                kind=ConstraintKind.POINT, lat=best.lat, lon=best.lon,
                radius_km=0.4, confidence=Confidence.HIGH,
                note=f"VPR match: {best.site_name or best.site_id}")]))

    out.extend(ev for _score, ev in accepted.values())
    return out


def _solar_timing(image_path: Path, report: CaseReport,
                  person_boxes: list[tuple[int, int, int, int]]
                  ) -> tuple[dict | None, list[Evidence]]:
    """Estimate when the photograph was taken, from the shadows in it.

    Uses elevation rather than bearing. A shadow's direction in the frame is
    meaningless without the camera's heading, which files almost never
    record; the ratio between an upright subject and its shadow needs no
    heading at all. The cost is that the sun reaches every height twice a
    day, so two candidate times are reported unless something else resolves
    which side of noon it was.
    """
    measurement = shadows.measure(image_path, person_boxes)
    if measurement is None:
        return None, []

    out: list[Evidence] = []
    payload: dict = {"measurement": measurement.as_dict()}

    detail = (f"Sun elevation {measurement.elevation_deg:.0f}° above the "
              f"horizon, from {measurement.source} "
              f"(shadow:subject ratio {measurement.ratio:.2f}). "
              "Elevation is measured rather than assumed: it needs no camera "
              "heading, unlike a shadow bearing.")
    if measurement.notes:
        detail += " " + " ".join(measurement.notes)

    # Location and date, if the case has established them.
    lat = lon = None
    if report.candidates:
        lat, lon = report.candidates[0].lat, report.candidates[0].lon
    when = None
    for ev in report.evidence:
        if ev.id == "meta.timestamp" and ev.raw.get("datetime_original"):
            from contextlib import suppress
            with suppress(Exception):
                when = datetime.fromisoformat(
                    str(ev.raw["datetime_original"]).replace("/", "-").strip())

    if lat is not None and when is not None:
        pair = solar.times_for_elevation(lat, lon, when, measurement.elevation_deg)
        if pair is None:
            peak = solar.max_elevation_on(lat, lon, when)
            detail += (f" The sun never rises above {peak:.0f}° at this "
                       f"location on the stated date, so the shadows and the "
                       "timestamp are inconsistent -- one of them is wrong.")
            payload["contradiction"] = True
        else:
            am, pm = pair
            payload["candidate_times_utc"] = [am.isoformat(), pm.isoformat()]
            detail += (f" On the stated date at this location the sun is at "
                       f"that height twice: {am:%H:%M} and {pm:%H:%M} UTC.")
    elif lat is not None:
        # No date: bracket the year to give an honest window.
        year = datetime.now(UTC).year
        spans = []
        for label, date in (("midsummer", datetime(year, 6, 21, tzinfo=UTC)),
                            ("equinox", datetime(year, 3, 20, tzinfo=UTC)),
                            ("midwinter", datetime(year, 12, 21, tzinfo=UTC))):
            pair = solar.times_for_elevation(lat, lon, date,
                                             measurement.elevation_deg)
            if pair:
                spans.append(f"{label} {pair[0]:%H:%M}/{pair[1]:%H:%M}")
            else:
                spans.append(f"{label}: sun never that high")
        payload["seasonal_window"] = spans
        detail += (" No capture date survived, so the clock time depends on "
                   "the season. At this location: " + "; ".join(spans)
                   + " (UTC).")
        # A sun height near the local maximum is only reachable for part of
        # the year, which dates the photograph without any metadata.
        window = solar.date_window_for_elevation(
            lat, lon, measurement.elevation_deg, year)
        if window:
            payload["date_window"] = [window[0].strftime("%d %b"),
                                      window[1].strftime("%d %b")]
            detail += (f" That elevation is only attainable at this latitude "
                       f"between {window[0]:%d %B} and {window[1]:%d %B}, "
                       "which dates the photograph to that part of the year.")

    out.append(Evidence(
        id="solar.elevation", analyzer="solar",
        title=f"Sun elevation {measurement.elevation_deg:.0f}° from shadows",
        detail=detail,
        confidence={"medium": Confidence.MEDIUM,
                    "low": Confidence.LOW}.get(measurement.confidence,
                                               Confidence.LOW),
        tags=["solar", "shadow", "time"], raw=payload))
    return payload, out


def analyze_image(image_path: Path, *, analyst: AnalystInput | None = None,
                  run_scene_model: bool = True, use_habitation_prior: bool = True,
                  auto_geocode: bool = True,
                  run_vpr: bool = True, vpr_area: str | None = None,
                  case_dir: Path | None = None, n_candidates: int = 8,
                  progress=None) -> CaseReport:
    """Full analysis of a single image. Returns a saved, complete CaseReport."""
    analyst = analyst or AnalystInput()
    SETTINGS.ensure_dirs()
    image_path = Path(image_path).resolve()

    def tick(msg: str):
        if progress:
            progress(msg)

    case_id = new_case_id(image_path)
    out_dir = Path(case_dir) if case_dir else SETTINGS.case_dir / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = CaseReport(
        case_id=case_id, created_utc=datetime.now(UTC),
        image=image_facts(image_path), offline_mode=SETTINGS.offline,
    )

    grid = world_grid()
    heatmaps: dict[str, np.ndarray] = {}

    # ---- metadata -------------------------------------------------------
    tick("metadata forensics")
    try:
        report.evidence.extend(metadata.analyze(image_path))
        report.analyzers_run.append("metadata")
    except Exception as exc:
        report.analyzers_skipped["metadata"] = str(exc)

    # ---- faces (guardrail) ----------------------------------------------
    tick("face detection")
    try:
        report.faces = faces.detect(image_path)
        report.analyzers_run.append("faces")
        if report.faces.count or report.faces.people_count:
            people = max(report.faces.people_count, report.faces.count)
            report.warnings.append(
                f"{people} person/people detected, {report.faces.count} with a "
                "visible face. This tool geolocates scenes and performs no "
                "facial recognition; detected faces are blurred in exported "
                "imagery. Note that people photographed from behind remain "
                "identifiable by clothing, build and companions, so the "
                "absence of visible faces does not make the image anonymous. "
                "Consider whether geolocating an image of identifiable people "
                "is within the scope of your authorisation."
            )
            if SETTINGS.blur_faces_in_exports:
                redacted = faces.write_redacted(
                    image_path, report.faces, out_dir / "image_redacted.jpg")
                if redacted:
                    report.faces.blurred_export = redacted.name
    except Exception as exc:
        report.analyzers_skipped["faces"] = str(exc)

    # ---- OCR ------------------------------------------------------------
    tick("text extraction (OCR)")
    try:
        report.evidence.extend(text_ocr.analyze(image_path))
        report.analyzers_run.append("text")
    except Exception as exc:
        report.analyzers_skipped["text"] = str(exc)

    # ---- scene model -----------------------------------------------------
    if run_scene_model:
        tick("scene model (StreetCLIP)")
        try:
            report.evidence.extend(clip_scene.analyze(image_path, heatmaps=heatmaps, grid=grid))
            report.analyzers_run.append("scene")
        except Exception as exc:
            report.analyzers_skipped["scene"] = str(exc)
    else:
        report.analyzers_skipped["scene"] = "disabled by caller"

    # ---- solar ------------------------------------------------------------
    tick("solar geometry")
    exif_utc, exif_local = _exif_datetime(report.evidence)
    when_utc = analyst.capture_datetime_utc or exif_utc
    date_only = analyst.capture_date or (None if when_utc else exif_local)
    try:
        solar_ev = solar.analyze(
            grid, when_utc=when_utc, date_only=date_only,
            shadow_azimuth=analyst.shadow_azimuth,
            sun_elevation=analyst.sun_elevation,
            object_height=analyst.object_height,
            shadow_length=analyst.shadow_length,
            heatmaps=heatmaps,
        )
        report.evidence.extend(solar_ev)
        if solar_ev:
            report.analyzers_run.append("solar")
    except Exception as exc:
        report.analyzers_skipped["solar"] = str(exc)

    # ---- analyst input ----------------------------------------------------
    report.evidence.extend(_analyst_evidence(analyst))

    # ---- fuse -------------------------------------------------------------
    # Two passes. The first establishes which countries are plausible; that
    # shortlist is what makes geocoding image text useful, since the same
    # shop name exists in fifty countries. The second pass folds the
    # resulting coordinates back in.
    tick("fusing evidence")
    post, grid = fusion.fuse(report.evidence, grid=grid, heatmaps=heatmaps,
                             use_habitation_prior=use_habitation_prior)
    report.top_countries = fusion.top_countries(post, grid)

    if auto_geocode and SETTINGS.net_allowed():
        tick("geocoding image text")
        shortlist = [cc for cc, share in report.top_countries[:5] if share > 0.02]
        try:
            geo_ev = textgeo.geocode_evidence(report.evidence, shortlist or None)
            geo_ev += textgeo.geocode_postcodes(report.evidence, shortlist or None)
        except Exception as exc:
            report.analyzers_skipped["geocode"] = str(exc)
            geo_ev = []
        if geo_ev:
            report.evidence.extend(geo_ev)
            report.analyzers_run.append("geocode")
            tick("re-fusing with geocoded text")
            post, grid = fusion.fuse(report.evidence, grid=grid, heatmaps=heatmaps,
                                     use_habitation_prior=use_habitation_prior)
            report.top_countries = fusion.top_countries(post, grid)
    elif auto_geocode:
        report.analyzers_skipped["geocode"] = "offline mode"

    # ---- facility gazetteer ------------------------------------------
    # "LIBRARY" cannot be geocoded, but it says the photo is a campus -- and
    # the coordinates of every campus in a country are a matter of record.
    # That turns "somewhere in Australia" into a list of real sites.
    if auto_geocode:
        try:
            sites_ev = _facility_evidence(report, analyst)
        except Exception as exc:
            report.analyzers_skipped["facility"] = str(exc)
            sites_ev = []
        if sites_ev:
            report.evidence.extend(sites_ev)
            report.analyzers_run.append("facility")
            tick("re-fusing with candidate sites")
            post, grid = fusion.fuse(report.evidence, grid=grid, heatmaps=heatmaps,
                                     use_habitation_prior=use_habitation_prior)
            report.top_countries = fusion.top_countries(post, grid)

    # ---- visual place recognition ---------------------------------------
    if run_vpr:
        tick("matching against reference imagery")
        try:
            vpr_ev = _vpr_evidence(image_path, report, vpr_area)
        except Exception as exc:
            report.analyzers_skipped["vpr"] = str(exc)
            vpr_ev = []
        if vpr_ev:
            report.evidence.extend(vpr_ev)
            report.analyzers_run.append("vpr")
            if any(e.constraints for e in vpr_ev):
                tick("re-fusing with the visual match")
                post, grid = fusion.fuse(report.evidence, grid=grid,
                                         heatmaps=heatmaps,
                                         use_habitation_prior=use_habitation_prior)
                report.top_countries = fusion.top_countries(post, grid)

    report.candidates = fusion.extract_candidates(
        post, grid, n=n_candidates, evidence=report.evidence)
    report.entropy_bits = fusion.posterior_entropy_bits(post)

    for msg in fusion.detect_contradictions(report.evidence, grid, heatmaps):
        report.warnings.append(msg)

    # ---- capture time from shadows, and metadata verification -----------
    tick("measuring shadows")
    try:
        person_boxes = faces.detect_people_boxes(image_path)
        solar_payload, solar_ev = _solar_timing(image_path, report, person_boxes)
    except Exception as exc:
        report.analyzers_skipped["solar_shadow"] = str(exc)
        solar_payload, solar_ev = None, []
    if solar_ev:
        report.evidence.extend(solar_ev)
        report.analyzers_run.append("solar")
    report.solar_timing = solar_payload

    tick("verifying metadata against content")
    try:
        report.metadata_verdict = metaverify.verify(
            report.evidence, grid, heatmaps,
            use_habitation_prior=use_habitation_prior).as_dict()
    except Exception as exc:
        report.analyzers_skipped["metaverify"] = str(exc)
    report.metadata_fields = metadata.full_dump(image_path)

    # A fabricated GPS tag outranks every other constraint by design, because
    # a genuine one deserves to. That makes an unflagged conflict the single
    # most dangerous output this tool can produce: the ranked answer sits on
    # the tag, looking authoritative, with the contradicting evidence buried.
    verdict = report.metadata_verdict or {}
    if verdict.get("verdict") == "conflict":
        report.warnings.insert(0, "METADATA CONFLICT. " + verdict["headline"]
                               + " " + verdict.get("detail", ""))

    report.credible_area_km2 = fusion.credible_region_km2(post, grid)
    band, note = fusion.describe_precision(report.credible_area_km2)
    report.precision_band = band
    report.precision_note = note

    if band in {"country", "unconstrained"}:
        report.warnings.append(
            f"Precision: {band.upper()}. 90% of the posterior covers "
            f"{report.credible_area_km2:,.0f} km2. {note}"
        )

    from .summary import build_summary
    report.summary = build_summary(
        report, shadow_azimuth=analyst.shadow_azimuth,
        capture_date=analyst.capture_date or date_only or when_utc)

    # ---- persist -----------------------------------------------------------
    np.save(out_dir / "posterior.npy", post.astype(np.float32))
    if not (out_dir / image_path.name).exists():
        shutil.copy2(image_path, out_dir / image_path.name)
    report.posterior_png = "posterior.png"
    from .viz import render_posterior_png
    render_posterior_png(post, grid, out_dir / "posterior.png")

    (out_dir / "case.json").write_text(report.model_dump_json(indent=2))
    tick("done")
    return report


def load_case(case_dir: Path) -> tuple[CaseReport, np.ndarray]:
    report = CaseReport.model_validate_json((case_dir / "case.json").read_text())
    post = np.load(case_dir / "posterior.npy")
    return report, post
