"""Metadata forensics.

First stop in any image investigation. Three classes of finding:

1. Direct location -- an intact EXIF/XMP GPS tag. Rare on anything that has
   passed through a social platform, and never trusted blindly: it is also
   the easiest field in the file to forge, so it is fused as evidence rather
   than treated as an answer.
2. Indirect location -- the UTC offset recorded alongside the local
   timestamp pins longitude to roughly one timezone band.
3. Provenance -- what device made it, what software touched it, and whether
   the metadata was stripped or rewritten. This tells the analyst how much
   to trust everything else.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..models import Confidence, ConstraintKind, Evidence, GeoConstraint

# Software strings that indicate the file was re-encoded rather than
# straight off a camera. Presence downgrades trust in every other tag.
EDITOR_HINTS = (
    "photoshop", "lightroom", "gimp", "affinity", "snapseed", "picsart",
    "capture one", "luminar", "darktable", "pixelmator", "canva",
)
PLATFORM_HINTS = ("whatsapp", "instagram", "facebook", "telegram", "signal", "twitter")


def _dms_to_deg(dms, ref: str | None) -> float | None:
    try:
        vals = [float(getattr(x, "num", x)) / float(getattr(x, "den", 1)) for x in dms]
        while len(vals) < 3:
            vals.append(0.0)
        deg = vals[0] + vals[1] / 60.0 + vals[2] / 3600.0
    except (TypeError, ValueError, ZeroDivisionError, AttributeError):
        return None
    if ref and ref.strip().upper() in {"S", "W"}:
        deg = -deg
    return deg


def _exiftool_dump(path: Path) -> dict[str, Any] | None:
    """Richer extraction when the exiftool binary is available locally."""
    exe = shutil.which("exiftool")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-json", "-n", "-G", "-a", "-u", str(path)],
            capture_output=True, timeout=30, check=False,
        )
        data = json.loads(out.stdout.decode("utf-8", "replace"))
        return data[0] if data else None
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError, OSError):
        return None


def _read_exifread(path: Path) -> dict[str, Any]:
    import exifread

    with path.open("rb") as fh:
        tags = exifread.process_file(fh, details=False)
    return {k: str(v) for k, v in tags.items()}


def _offset_to_lon_constraint(offset_str: str) -> GeoConstraint | None:
    """Convert a recorded UTC offset into a longitude band.

    One hour of offset is 15 degrees of longitude. Political timezone borders
    wander considerably from the solar meridian (China is a single zone
    spanning ~60 degrees), so the band is generously soft.
    """
    s = offset_str.strip()
    if not s:
        return None
    try:
        sign = -1 if s.startswith("-") else 1
        body = s.lstrip("+-")
        parts = body.split(":")
        hours = int(parts[0]) + (int(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        hours *= sign
    except (ValueError, IndexError):
        return None
    if not -12.5 <= hours <= 14.5:
        return None
    centre = hours * 15.0
    return GeoConstraint(
        kind=ConstraintKind.LON_BAND,
        lon_min=centre - 15.0, lon_max=centre + 15.0,
        softness_deg=18.0, floor=0.05,
        confidence=Confidence.MEDIUM,
        note=f"UTC offset {s} implies a longitude band centred near {centre:+.0f} deg",
    )


def analyze(path: Path) -> list[Evidence]:
    out: list[Evidence] = []
    et = _exiftool_dump(path)
    try:
        raw = _read_exifread(path)
    except Exception:
        raw = {}

    merged: dict[str, Any] = {}
    if et:
        merged.update({k: v for k, v in et.items() if not k.startswith("File:")})
    merged.update(raw)

    # ---- 1. direct GPS -------------------------------------------------
    lat = lon = None
    if et and et.get("EXIF:GPSLatitude") is not None:
        try:
            lat = float(et["EXIF:GPSLatitude"])
            lon = float(et["EXIF:GPSLongitude"])
        except (TypeError, ValueError, KeyError):
            lat = lon = None
    if lat is None:
        import exifread
        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False)
        gl, gr = tags.get("GPS GPSLatitude"), tags.get("GPS GPSLatitudeRef")
        go, gor = tags.get("GPS GPSLongitude"), tags.get("GPS GPSLongitudeRef")
        if gl and go:
            lat = _dms_to_deg(gl.values, str(gr) if gr else None)
            lon = _dms_to_deg(go.values, str(gor) if gor else None)

    if lat is not None and lon is not None and (abs(lat) > 1e-6 or abs(lon) > 1e-6):
        alt = merged.get("EXIF:GPSAltitude")
        out.append(Evidence(
            id="meta.gps",
            analyzer="metadata",
            title="Embedded GPS coordinates",
            detail=(f"EXIF GPS tag present: {lat:.6f}, {lon:.6f}"
                    + (f" at {alt} m altitude" if alt else "")
                    + ". Treat as a strong lead, not proof -- GPS tags are "
                      "trivially editable and are fused as evidence here."),
            confidence=Confidence.CERTAIN,
            tags=["gps", "direct"],
            raw={"lat": lat, "lon": lon, "altitude": alt},
            constraints=[GeoConstraint(
                kind=ConstraintKind.POINT, lat=lat, lon=lon, radius_km=2.0,
                confidence=Confidence.CERTAIN, note="EXIF GPS tag",
            )],
        ))
    else:
        out.append(Evidence(
            id="meta.gps_absent",
            analyzer="metadata",
            title="No embedded GPS coordinates",
            detail="No usable GPS tag. Common after upload to any major "
                   "platform, all of which strip location metadata.",
            confidence=Confidence.HIGH, tags=["gps"],
        ))

    # ---- 2. timestamps and UTC offset ----------------------------------
    dt_original = (merged.get("EXIF:DateTimeOriginal")
                   or merged.get("EXIF DateTimeOriginal")
                   or merged.get("Image DateTime"))
    offset = (merged.get("EXIF:OffsetTimeOriginal")
              or merged.get("EXIF OffsetTimeOriginal")
              or merged.get("EXIF:OffsetTime"))
    if dt_original:
        ev = Evidence(
            id="meta.timestamp", analyzer="metadata",
            title="Capture timestamp",
            detail=f"DateTimeOriginal = {dt_original}"
                   + (f", UTC offset {offset}" if offset else
                      ", no UTC offset recorded (local time only)"),
            confidence=Confidence.HIGH, tags=["time"],
            raw={"datetime_original": str(dt_original), "offset": str(offset or "")},
        )
        if offset:
            c = _offset_to_lon_constraint(str(offset))
            if c:
                ev.constraints.append(c)
        out.append(ev)

    # ---- 3. provenance --------------------------------------------------
    make = merged.get("EXIF:Make") or merged.get("Image Make") or ""
    model = merged.get("EXIF:Model") or merged.get("Image Model") or ""
    software = str(merged.get("EXIF:Software") or merged.get("Image Software") or "")
    lens = merged.get("EXIF:LensModel") or ""

    if make or model:
        out.append(Evidence(
            id="meta.device", analyzer="metadata",
            title="Capture device",
            detail=f"{make} {model}".strip() + (f" | lens: {lens}" if lens else ""),
            confidence=Confidence.HIGH, tags=["device"],
            raw={"make": str(make), "model": str(model), "lens": str(lens)},
        ))

    lowered = software.lower()
    if software:
        kind = "editing software" if any(h in lowered for h in EDITOR_HINTS) else "software"
        out.append(Evidence(
            id="meta.software", analyzer="metadata",
            title=f"Processed by {kind}",
            detail=f"Software tag: {software}. Pixel content and any "
                   "surviving metadata may have been altered.",
            confidence=Confidence.MEDIUM, tags=["provenance"],
            raw={"software": software},
        ))

    filename = path.name.lower()
    if any(h in filename for h in PLATFORM_HINTS) or filename.startswith("img-"):
        out.append(Evidence(
            id="meta.platform_filename", analyzer="metadata",
            title="Filename suggests messaging-platform origin",
            detail=f"Filename '{path.name}' matches a messaging or social "
                   "platform naming convention; original metadata was almost "
                   "certainly stripped at upload.",
            confidence=Confidence.LOW, tags=["provenance"],
        ))

    tag_count = len([k for k in merged if not k.startswith("File:")])
    if tag_count < 5:
        out.append(Evidence(
            id="meta.stripped", analyzer="metadata",
            title="Metadata appears stripped",
            detail=f"Only {tag_count} metadata fields present. The image has "
                   "likely been re-encoded or scrubbed. Scene-content analysis "
                   "is the only viable route.",
            confidence=Confidence.MEDIUM, tags=["provenance"],
        ))

    if not shutil.which("exiftool"):
        out.append(Evidence(
            id="meta.exiftool_missing", analyzer="metadata",
            title="exiftool not installed (reduced metadata coverage)",
            detail="Falling back to pure-Python EXIF parsing. Install with "
                   "`brew install exiftool` for XMP, IPTC, MakerNotes and "
                   "vendor-specific tags.",
            confidence=Confidence.LOW, tags=["tooling"],
        ))

    return out

def full_dump(path: Path) -> dict[str, Any]:
    """Every metadata field the file carries, for the operator to inspect.

    The selective evidence items above answer "what does this tell us about
    location"; this answers "what is actually in the file". Operators need
    both -- a lens model or an editing-software tag is often what exposes a
    doctored image, and no analyzer can anticipate which field will matter.
    """
    out: dict[str, Any] = {}

    # exiftool reads XMP, IPTC, MakerNotes and vendor blocks that neither
    # Pillow nor exifread expose. Used when present, but never required.
    dumped = _exiftool_dump(path)
    if dumped:
        for key, value in dumped.items():
            if key.startswith(("SourceFile", "ExifTool")):
                continue
            out[key] = value
        return out

    try:
        from PIL import ExifTags, Image

        with Image.open(path) as im:
            out["_Image.Format"] = im.format
            out["_Image.Mode"] = im.mode
            out["_Image.Size"] = f"{im.width}x{im.height}"
            exif = im.getexif()
            for tag_id, value in (exif or {}).items():
                name = ExifTags.TAGS.get(tag_id, f"Tag{tag_id}")
                out[name] = _stringify(value)
            for ifd_name, ifd_id in (("GPS", 0x8825), ("Exif", 0x8769)):
                try:
                    ifd = exif.get_ifd(ifd_id)
                except Exception:
                    continue
                table = ExifTags.GPSTAGS if ifd_name == "GPS" else ExifTags.TAGS
                for tag_id, value in (ifd or {}).items():
                    name = table.get(tag_id, f"Tag{tag_id}")
                    out[f"{ifd_name}.{name}"] = _stringify(value)
    except Exception:
        pass
    return out


def _stringify(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "replace")[:200]
        except Exception:
            return f"<{len(value)} bytes>"
    text = str(value)
    return text[:300] if len(text) > 300 else text


def camera_heading(path: Path) -> float | None:
    """Compass bearing the camera faced, if the file records one.

    Rare -- most phones omit it and most re-encoders strip it -- but when
    present it removes the single obstacle to dating a photograph from its
    shadows: a shadow's bearing in the frame is only meaningful once the
    frame's own orientation is known.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825) or {}
    except Exception:
        return None
    # 17 = GPSImgDirection, 16 = GPSImgDirectionRef ("T" true / "M" magnetic)
    raw = gps.get(17)
    if raw is None:
        return None
    try:
        heading = float(raw)
    except (TypeError, ValueError):
        return None
    return heading % 360.0

