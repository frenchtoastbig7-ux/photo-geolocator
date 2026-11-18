"""Recovering which way the camera was pointing.

Two independent routes, useful for different images and stronger together.

**From the sun.** If the place and the moment are known, the sun's azimuth
follows from astronomy, and a shadow on the ground runs directly away from
it. Comparing that true bearing with the shadow's direction *in the frame*
yields the camera's heading.

The comparison cannot be done naively. A shadow's apparent direction in an
image is not its ground azimuth minus the heading: perspective rotates it by
an amount that depends on where in the frame it falls. Measured against a
correct pinhole projection, that assumption is wrong by up to 87 degrees --
worse than useless, since it would look plausible while pointing the wrong
way. The shadow's endpoints are therefore back-projected onto the ground
plane and the direction measured there, which is exact.

Camera height cancels in that back-projection: it scales both endpoints
equally and so leaves their direction unchanged. Only the focal length and
the horizon's position in frame are needed, and both are recoverable.

**From the ground.** Once a site is identified, the streets, platforms and
building edges around it have known bearings in OpenStreetMap. Strong
parallel lines receding in the image give the direction the camera looks
along, up to a 180-degree ambiguity, and matching those against the mapped
bearings resolves the heading without needing the sun at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# A phone at its default zoom spans roughly 65-70 degrees horizontally, which
# puts the focal length near 0.7 x image width. Used only when the file
# records nothing better.
DEFAULT_FOV_DEG = 67.0


@dataclass
class HeadingEstimate:
    heading_deg: float
    method: str
    confidence: str
    ambiguity_deg: float = 0.0
    """0 for a unique solution; 180 when the evidence cannot tell forwards
    from backwards, as parallel ground lines cannot."""
    detail: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"heading_deg": round(self.heading_deg, 1), "method": self.method,
                "confidence": self.confidence,
                "ambiguity_deg": self.ambiguity_deg,
                "detail": self.detail, "notes": self.notes}


def focal_length_px(path: Path, width_px: int) -> tuple[float, str]:
    """Focal length in pixels, from the file if it says, else assumed."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            ifd = exif.get_ifd(0x8769) or {}
        # 41989 = FocalLengthIn35mmFilm; a 35 mm frame is 36 mm wide.
        f35 = ifd.get(41989)
        if f35:
            return width_px * float(f35) / 36.0, f"EXIF 35mm-equivalent {float(f35):.0f}mm"
    except Exception:
        pass
    return (width_px / (2.0 * math.tan(math.radians(DEFAULT_FOV_DEG / 2.0))),
            f"assumed {DEFAULT_FOV_DEG:.0f}° field of view")


def ground_direction_deg(foot_uv, tip_uv, f_px: float, horizon_v: float,
                         cx: float) -> float | None:
    """Bearing of a ground segment, clockwise from the optical axis."""
    pts = []
    for u, v in (foot_uv, tip_uv):
        dv = v - horizon_v
        if dv <= 1e-6:
            return None
        depth = f_px / dv
        lateral = (u - cx) * depth / f_px
        pts.append((lateral, depth))
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360.0


def heading_from_sun(relative_shadow_deg: float, sun_azimuth_deg: float
                     ) -> float:
    """Camera heading from a shadow's ground bearing relative to the axis.

    A shadow points directly away from the sun, so its true bearing is the
    solar azimuth plus 180 degrees. The heading is whatever makes the
    measured relative bearing agree with that.
    """
    true_shadow = (sun_azimuth_deg + 180.0) % 360.0
    return (true_shadow - relative_shadow_deg) % 360.0


def _segments(raw) -> list[tuple[float, float, float, float]]:
    """Normalise HoughLinesP output across OpenCV versions.

    OpenCV 4 returns each segment wrapped in a length-1 array; OpenCV 5
    returns the four endpoints directly. Unpacking one shape blindly raises
    on the other.
    """
    out: list[tuple[float, float, float, float]] = []
    for seg in raw:
        vals = np.asarray(seg).reshape(-1)
        if vals.size >= 4:
            out.append((float(vals[0]), float(vals[1]),
                        float(vals[2]), float(vals[3])))
    return out


def estimate_horizon(path: Path) -> tuple[float, str]:
    """Height of the horizon line in the image, in pixels from the top.

    This matters more than any other parameter. Assuming the horizon sits at
    the image centre -- i.e. that the camera was perfectly level -- costs
    about 22 degrees of heading error on a photograph tilted 10 degrees down,
    which is an ordinary way to hold a phone. Locating it properly brings
    that back to a fraction of a degree.

    Parallel lines on the ground converge at a point on the horizon, so the
    intersections of strong receding edges cluster at its height. The median
    intersection is used, being robust to the many line pairs that are not
    actually parallel in the world.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, "unreadable"
    h, w = img.shape[:2]
    edges = cv2.Canny(img, 60, 180, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                           minLineLength=int(min(h, w) * 0.15), maxLineGap=14)
    if segs is None or len(segs) < 4:
        return h / 2.0, "assumed level camera (no usable lines)"

    lines = []
    for x1, y1, x2, y2 in _segments(segs)[:150]:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        # Only obliques can converge; horizontals and verticals cannot fix a
        # vanishing point's height.
        if 12.0 < ang < 78.0 or 102.0 < ang < 168.0:
            lines.append((x1, y1, x2, y2))
    if len(lines) < 4:
        return h / 2.0, "assumed level camera (too few oblique lines)"

    ys: list[float] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i]
            x3, y3, x4, y4 = lines[j]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-6:
                continue
            num = ((x1 * y2 - y1 * x2) * (y3 - y4)
                   - (y1 - y2) * (x3 * y4 - y3 * x4))
            y = num / den
            # A vanishing point far outside the frame is a numerical artefact
            # of two nearly-parallel image lines, not a real horizon.
            if -h < y < 2 * h:
                ys.append(y)
    if len(ys) < 6:
        return h / 2.0, "assumed level camera (no vanishing point)"
    return float(np.median(ys)), f"vanishing point of {len(lines)} ground lines"


def dominant_ground_lines(path: Path, max_lines: int = 200
                          ) -> list[float]:
    """Bearings, relative to the optical axis, of strong receding ground lines.

    Kerbs, platform edges and building bases are parallel in the world and so
    converge in the image. Their shared vanishing point gives the direction
    the camera looks along -- but only modulo 180 degrees, since a line
    running away from the camera and one running toward it are the same line.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape[:2]
    edges = cv2.Canny(img, 60, 180, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90,
                           minLineLength=int(min(h, w) * 0.12), maxLineGap=12)
    if segs is None:
        return []

    angles: list[float] = []
    for x1, y1, x2, y2 in _segments(segs)[:max_lines]:
        # Ignore near-horizontal and near-vertical lines: the first are
        # usually the horizon or roof lines, the second are uprights, and
        # neither constrains a ground bearing.
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        if 15.0 < ang < 75.0 or 105.0 < ang < 165.0:
            angles.append(ang)
    return angles


# Shadows cast by the same sun are parallel on the ground. Back-projected
# bearings that disagree by more than this are not both shadows, and no
# heading derived from them can be trusted.
AGREEMENT_TOLERANCE_DEG = 20.0


def relative_shadow_bearing(endpoints, f_px: float, horizon_v: float,
                            cx: float) -> tuple[float, float, int] | None:
    """Consensus ground bearing of the measured shadows.

    Returns (bearing, spread, count) relative to the optical axis, or None if
    the shadows do not agree. The agreement check is the point: a segmenter
    that latches onto a dark doorway instead of a shadow produces a confident
    bearing pointing the wrong way, and only cross-checking against another
    shadow reveals it.
    """
    bearings: list[float] = []
    for foot_u, foot_v, tip_u, tip_v in endpoints:
        b = ground_direction_deg((foot_u, foot_v), (tip_u, tip_v),
                                 f_px, horizon_v, cx)
        if b is not None:
            bearings.append(b)
    if not bearings:
        return None
    if len(bearings) == 1:
        return bearings[0], 0.0, 1

    ang = np.radians(bearings)
    mean = math.degrees(math.atan2(np.sin(ang).mean(),
                                   np.cos(ang).mean())) % 360.0
    spread = max(min(abs(b - mean), 360.0 - abs(b - mean)) for b in bearings)
    if spread > AGREEMENT_TOLERANCE_DEG:
        return None
    return mean, spread, len(bearings)


def estimate(path: Path, endpoints, *, sun_azimuth_deg: float | None,
             width_px: int) -> HeadingEstimate | None:
    """Camera heading from shadows, given where the sun was."""
    if not endpoints or sun_azimuth_deg is None:
        return None

    f_px, f_note = focal_length_px(path, width_px)
    horizon_v, hz_note = estimate_horizon(path)
    consensus = relative_shadow_bearing(endpoints, f_px, horizon_v, width_px / 2.0)
    if consensus is None:
        return HeadingEstimate(
            heading_deg=float("nan"), method="solar-shadow",
            confidence="rejected",
            detail=("The measured shadows do not run parallel once projected "
                    "onto the ground, so at least one is not a shadow. No "
                    "heading is derived; a bearing from a mis-segmented dark "
                    "region would look just as confident and point the wrong "
                    "way."))

    bearing, spread, count = consensus
    heading = heading_from_sun(bearing, sun_azimuth_deg)

    notes = [f"Focal length: {f_note}.", f"Horizon: {hz_note}."]
    # Tilt dominates the error budget: assuming a level camera costs about 22
    # degrees on a photo tilted 10 degrees down, against 5 degrees for a
    # focal length that is wrong by a third.
    if hz_note.startswith("assumed"):
        confidence = "low"
        notes.append("With the horizon assumed rather than measured, expect "
                     "errors of tens of degrees if the camera was tilted.")
    elif f_note.startswith("assumed"):
        confidence = "medium"
        notes.append("Focal length assumed; expect a few degrees of error.")
    else:
        confidence = "high"
    if count == 1:
        notes.append("Only one shadow was measurable, so nothing cross-checks it.")
        confidence = "low" if confidence == "high" else confidence

    return HeadingEstimate(
        heading_deg=heading, method="solar-shadow", confidence=confidence,
        detail=(f"Shadows run {bearing:.0f}° from the optical axis once "
                f"back-projected onto the ground; with the sun at "
                f"{sun_azimuth_deg:.0f}° the camera faced {heading:.0f}°. "
                f"Agreement across {count} shadow(s): {spread:.0f}°."),
        notes=notes)


def match_against_features(candidates: list[tuple[float, str]],
                           feature_bearings: list[tuple[float, str]],
                           tolerance_deg: float = 12.0
                           ) -> tuple[float, str, str, int] | None:
    """Pick the candidate heading that lines up with the streets on the map.

    The sun leaves an ambiguity that astronomy cannot resolve: it reaches the
    same elevation once climbing and once descending, and the two moments
    imply different headings. The ground does resolve it. A photographer
    standing in a street is usually looking along it, so the candidate that
    aligns with a mapped way is the one to keep.

    Bearings are compared modulo 180, since a way has no inherent direction.
    Returns (heading, label, matched_feature,count) or None when no
    candidate aligns with anything, which is the honest outcome for an open
    space with no mapped structure.
    """
    if not candidates or not feature_bearings:
        return None

    best = None
    for heading, label in candidates:
        folded = heading % 180.0
        aligned = [name for bearing, name in feature_bearings
                   if min(abs(folded - bearing),
                          180.0 - abs(folded - bearing)) <= tolerance_deg]
        if not aligned:
            continue
        if best is None or len(aligned) > best[3]:
            common = max(set(aligned), key=aligned.count)
            best = (heading, label, common, len(aligned))
    return best
