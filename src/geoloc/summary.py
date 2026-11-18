"""Analyst-facing intelligence brief.

The evidence board records what each analyzer found; this turns it into the
digest an analyst actually wants at the top of a case file: what the place
probably is, where, when, what is visible, how much to trust the file, and
what to do next.

Everything here is derived from evidence already in the report. It states
what is *not* known as plainly as what is -- an absent capture time is a fact
about the case, not a blank to be skipped over.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from .models import CaseReport


def _ev(report: CaseReport, eid: str):
    return report.evidence_by_id(eid)


def _scene_axis(report: CaseReport, axis: str) -> tuple[str, float] | None:
    ev = _ev(report, f"clip.{axis}")
    if not ev:
        return None
    ranked = ev.raw.get("ranked") or []
    if not ranked:
        return None
    label, prob = ranked[0]
    return str(label), float(prob)


def _observed_text(report: CaseReport) -> list[str]:
    ev = _ev(report, "ocr.text")
    if not ev:
        return []
    return [str(b.get("text", "")).strip()
            for b in ev.raw.get("blocks", []) if str(b.get("text", "")).strip()]


def estimate_time_of_day(report: CaseReport, *, shadow_azimuth: float | None,
                         capture_date: datetime | None) -> dict[str, Any]:
    """Best available statement about when the photo was taken.

    Three tiers, in descending order of rigour: a recorded EXIF timestamp; a
    solar solution from a measured shadow at the leading candidate location;
    or nothing, said plainly.
    """
    out: dict[str, Any] = {"method": "none", "text": "Not determinable from this image."}

    # Measured sun elevation is the fallback that needs no camera heading and
    # no surviving metadata, so it is prepared first and used if nothing
    # better exists.
    measured = None
    if report.solar_timing:
        m = report.solar_timing.get("measurement") or {}
        elev = m.get("elevation_deg")
        if elev is not None:
            offset_h = round(report.candidates[0].lon / 15.0) if report.candidates else 0
            if report.solar_timing.get("contradiction"):
                measured = {
                    "method": "shadow_elevation",
                    "elevation_deg": elev,
                    "text": (f"Shadows put the sun {elev:.0f}° above the "
                             "horizon, which is higher than it ever reaches "
                             "at this location on the recorded date. The "
                             "timestamp and the image disagree.")}
            elif report.solar_timing.get("candidate_times_utc"):
                am, pm = report.solar_timing["candidate_times_utc"]
                am_t, pm_t = am[11:16], pm[11:16]
                measured = {
                    "method": "shadow_elevation",
                    "elevation_deg": elev,
                    "candidates_utc": [am_t, pm_t],
                    "text": (f"Sun measured at {elev:.0f}° above the horizon "
                             f"from shadows. At this location and date that "
                             f"occurs twice: {am_t} or {pm_t} UTC "
                             f"(about {(int(am_t[:2])+offset_h)%24:02d}:{am_t[3:]} "
                             f"or {(int(pm_t[:2])+offset_h)%24:02d}:{pm_t[3:]} "
                             "local). The sun passes each height once "
                             "climbing and once descending, so morning and "
                             "afternoon cannot be separated by elevation "
                             "alone.")}
            else:
                window = report.solar_timing.get("seasonal_window") or []
                measured = {
                    "method": "shadow_elevation",
                    "elevation_deg": elev,
                    "text": (f"Sun measured at {elev:.0f}° above the horizon "
                             "from shadows. No capture date survived, so the "
                             "clock time depends on season"
                             + (": " + "; ".join(window) + " UTC."
                                if window else "."))}
                hd = report.solar_timing.get("heading_deg")
                if hd is not None:
                    compass = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S",
                               "SSW","SW","WSW","W","WNW","NW","NNW"][
                        int((hd + 11.25) % 360 // 22.5)]
                    measured["camera_heading_deg"] = hd
                    measured["text"] += (
                        f" Camera faced {hd:.0f}° ({compass}), from the "
                        "shadow geometry cross-checked against "
                        f"{report.solar_timing.get('heading_resolved_by','the street layout')}"
                        f"; that also resolves the {report.solar_timing.get('resolved_time','')} "
                        "solution.")
                elif report.solar_timing.get("heading_candidates"):
                    opts = ", ".join(
                        f"{c['heading_deg']:.0f}\u00b0 ({c['time']})"
                        for c in report.solar_timing["heading_candidates"])
                    measured["text"] += (f" Camera heading is {opts} depending "
                                         "on which side of noon.")
                dw = report.solar_timing.get("date_window")
                if dw:
                    measured["date_window"] = dw
                    measured["text"] += (f" That height is only reachable at "
                                         f"this latitude between {dw[0]} and "
                                         f"{dw[1]}, which dates the image to "
                                         "that part of the year.")
            if m.get("notes"):
                measured["text"] += " " + " ".join(m["notes"])
            measured["confidence"] = m.get("confidence", "low")

    meta = _ev(report, "meta.timestamp")
    if meta:
        raw = meta.raw.get("datetime_original", "")
        offset = meta.raw.get("offset", "")
        out = {"method": "exif",
               "local_time": str(raw),
               "utc_offset": str(offset or ""),
               "text": (f"Recorded by the camera as {raw}"
                        + (f" (UTC{offset})" if offset else
                           " (no UTC offset recorded, so local clock time)")
                        + ". Editable, so corroborate against shadows if it matters.")}
        if measured:
            out["shadow_check"] = measured["text"]
        return out

    if shadow_azimuth is not None and report.candidates and capture_date is not None:
        from .analyzers.solar import local_solar_time, solar_noon_utc

        top = report.candidates[0]
        solved = local_solar_time(top.lat, top.lon, capture_date, shadow_azimuth)
        if solved:
            instant, elevation, err = solved
            noon = solar_noon_utc(top.lat, top.lon, capture_date)
            offset_h = round(top.lon / 15.0)
            local = instant + timedelta(hours=offset_h)
            rel = ""
            if noon:
                delta = (instant - noon).total_seconds() / 3600.0
                rel = (f", about {abs(delta):.1f} h "
                       f"{'after' if delta > 0 else 'before'} local solar noon")
            out = {"method": "solar",
                   "utc": instant.isoformat(),
                   "local_approx": local.strftime("%H:%M"),
                   "sun_elevation_deg": round(elevation, 1),
                   "bearing_error_deg": round(err, 1),
                   "text": (f"Approximately {local.strftime('%H:%M')} local "
                            f"({instant.strftime('%H:%M')} UTC){rel}, with the "
                            f"sun {elevation:.0f}° above the horizon. Solved "
                            f"from the shadow bearing at candidate #1; if that "
                            f"candidate is wrong, so is this.")}
    return measured or out


def describe_scene(report: CaseReport) -> str:
    """One-paragraph description of what the photograph shows.

    Scene-model output is always attributed and never asserted. A softmax
    over a closed prompt list is not calibrated confidence -- it returns a
    high number for the best of the offered options whether or not any of
    them is right. Measured on a Gold Coast university campus, this model
    reported "hot desert" vegetation and "sub saharan african housing" at
    probabilities above 0.55. Reporting that as description would send an
    analyst to the wrong continent, so the model's readings are labelled as
    the model's, with their scores shown, and the text OCR actually read is
    presented separately as the one part that is directly observed.
    """
    readings: list[str] = []
    for axis, label in (("setting", "setting"), ("architecture", "built form"),
                        ("biome", "vegetation"), ("season", "season"),
                        ("utility", "utilities")):
        got = _scene_axis(report, axis)
        if got:
            readings.append(f"{label}: {got[0]} ({got[1] * 100:.0f}%)")

    parts: list[str] = []
    text = _observed_text(report)
    if text:
        parts.append("Text visible in the image: "
                     + "; ".join(f"\u201c{t}\u201d" for t in text[:8]) + ".")
    else:
        parts.append("No legible text was recovered from the image.")

    if readings:
        parts.append("Scene model readings (unreliable on their own, and "
                     "frequently wrong on the finer axes -- verify by eye): "
                     + "; ".join(readings) + ".")
    return " ".join(parts)


def provenance_notes(report: CaseReport) -> list[str]:
    notes: list[str] = []
    for eid, prefix in (("meta.device", "Camera"), ("meta.software", "Processing"),
                        ("meta.stripped", "Metadata"),
                        ("meta.platform_filename", "Filename")):
        ev = _ev(report, eid)
        if ev:
            notes.append(f"{prefix}: {ev.detail.splitlines()[0]}")
    if not _ev(report, "meta.gps"):
        notes.append("No GPS tag: location must be established from content alone.")
    return notes


def next_steps(report: CaseReport) -> list[str]:
    """What an analyst should do next, given how the case actually landed."""
    steps: list[str] = []
    band = report.precision_band

    facility = next((e for e in report.evidence if e.analyzer == "facility"), None)
    if facility:
        n = facility.raw.get("site_count", 0)
        steps.append(
            f"Check the {n} candidate sites against satellite or street-level "
            "imagery. The distinguishing features in this photo are the ones "
            "to look for: paving patterns, building signage, roof shapes.")
    if band in {"country", "unconstrained"}:
        steps.append(
            "Do not treat the ranked list as leads. Look for any additional "
            "readable text, a vehicle registration, a business name or a "
            "distinctive structure, and re-run -- one legible proper noun is "
            "worth more than every model in this pipeline.")
    if not _ev(report, "meta.timestamp"):
        steps.append(
            "No capture time survived. If a shadow is visible, measure its "
            "bearing from true north and re-run with --shadow-azimuth to "
            "constrain both place and time.")
    if report.faces.count or report.faces.people_count:
        people = max(report.faces.people_count, report.faces.count)
        steps.append(
            f"{people} person/people present. Confirm your authorisation "
            "covers imagery of identifiable people before circulating this "
            "case; exported imagery is redacted by default.")
    if not steps:
        steps.append("Corroborate the leading candidate against independent "
                     "imagery before relying on it.")
    return steps


def build_summary(report: CaseReport, *, shadow_azimuth: float | None = None,
                  capture_date: datetime | None = None) -> dict[str, Any]:
    top = report.candidates[0] if report.candidates else None
    trustworthy = report.precision_band in {"pinpoint", "locality", "regional"}

    # A disputed tag must not be reported as the answer. A genuine GPS tag
    # outranks everything else by design, so a fabricated one lands at the
    # top of the ranking looking authoritative; stating it plainly here is
    # the difference between a warning an operator may scroll past and a
    # headline they cannot.
    mv = report.metadata_verdict or {}
    if mv.get("verdict") == "conflict" and mv.get("content_best"):
        tag_lat, tag_lon = mv["gps"]
        best_lat, best_lon = mv["content_best"]
        return {
            "assessment": (
                f"DISPUTED. The file is tagged {tag_lat:.5f}, {tag_lon:.5f}, "
                f"but the image's own content points to {best_lat:.3f}, "
                f"{best_lon:.3f} — {mv['distance_km']:,.0f} km away. Do not "
                "report either as the location until the discrepancy is "
                "resolved."),
            "location": "DISPUTED — metadata and image content disagree",
            "coordinates": (f"tagged {tag_lat:.5f}, {tag_lon:.5f}  |  "
                            f"content {best_lat:.3f}, {best_lon:.3f}"),
            "countries": [f"{cc} {share * 100:.0f}%"
                          for cc, share in report.top_countries[:4]],
            "metadata_conflict": mv,
            "time_of_day": estimate_time_of_day(
                report, shadow_azimuth=shadow_azimuth, capture_date=capture_date),
            "people_present": max(report.faces.people_count, report.faces.count),
            "faces_visible": report.faces.count,
            "precision_band": report.precision_band,
            "next_steps": [
                "Resolve the metadata conflict first: everything downstream "
                "of a fabricated tag is unreliable.",
                "Work the image content independently — text, signage and "
                "structures — and treat the tag as a claim to be tested.",
            ],
        }

    if top and trustworthy:
        location = top.label
        coords = f"{top.lat:.5f}, {top.lon:.5f}"
        assessment = (f"Most likely {top.label}, carrying "
                      f"{top.score * 100:.1f}% of the posterior.")
    elif top:
        location = f"{top.country or 'unknown'} (country level only)"
        coords = f"{top.lat:.5f}, {top.lon:.5f} — indicative only"
        assessment = ("Not localised. The evidence supports a country at best; "
                      "the named candidates are not evidence-backed.")
    else:
        location, coords = "Undetermined", "—"
        assessment = "No candidate location could be derived from this image."

    return {
        "assessment": assessment,
        "location": location,
        "coordinates": coords,
        "precision_band": report.precision_band,
        "credible_area_km2": report.credible_area_km2,
        "countries": [f"{cc} {share * 100:.0f}%" for cc, share in report.top_countries[:4]],
        "time_of_day": estimate_time_of_day(
            report, shadow_azimuth=shadow_azimuth, capture_date=capture_date),
        "description": describe_scene(report),
        "observed_text": _observed_text(report),
        "provenance": provenance_notes(report),
        "people_present": max(report.faces.people_count, report.faces.count),
        "faces_visible": report.faces.count,
        "next_steps": next_steps(report),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
