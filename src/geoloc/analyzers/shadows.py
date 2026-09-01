"""Measuring the sun from shadows in the frame.

Dating a photograph from its shadows normally founders on one obstacle: a
shadow's bearing within the image says nothing about its bearing on the
ground until the camera's own orientation is known, and almost no file
records that. Every earlier attempt in this project therefore required the
analyst to supply a compass bearing by hand.

Sun *elevation* escapes that problem entirely. The ratio between an upright
object and the shadow it casts is a property of the sun's height, not of
where the photographer was standing:

    elevation = atan(object_height / shadow_length)

Both quantities can be measured in pixels, and the ratio survives the
conversion, so no heading is needed. Elevation plus a date and a latitude
gives the time of day -- with an unavoidable morning/afternoon ambiguity,
since the sun passes the same height twice.

The measurement is only as good as its assumptions: the object must be
upright, the ground beneath it flat, and the shadow must not run toward or
away from the camera, where foreshortening compresses it and inflates the
apparent elevation. Those conditions are checked where possible and reported
where not, because a confidently wrong time is worse than no time at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# An upright adult, used as the reference object. Only the ratio matters, so
# the absolute value cancels -- but it anchors the plausibility checks.
ADULT_HEIGHT_M = 1.7

# A shadow much shorter than a tenth of the object implies the sun is within
# a few degrees of vertical, which happens only in the tropics near noon;
# much longer than ten times implies near-sunset. Outside that band the
# measurement is far more likely to be a detection error than a real sun.
MIN_RATIO = 0.10
MAX_RATIO = 12.0

# Foreshortening: a shadow pointing nearly straight up or down the frame is
# running toward or away from the camera and its length is not trustworthy.
FORESHORTEN_GUARD_DEG = 25.0


@dataclass
class ShadowMeasurement:
    elevation_deg: float
    ratio: float
    image_bearing_deg: float
    """Direction the shadow runs, measured clockwise from image-up. Not a
    compass bearing: it is only meaningful once the camera heading is known."""
    confidence: str
    source: str
    notes: list[str] = field(default_factory=list)
    endpoints: list[tuple[float, float, float, float]] = field(default_factory=list)
    """(foot_u, foot_v, tip_u, tip_v) per measured shadow. Kept because the
    camera heading is recovered by back-projecting these onto the ground
    plane, which needs the actual image positions, not just a bearing."""

    def as_dict(self) -> dict[str, Any]:
        return {"elevation_deg": round(self.elevation_deg, 1),
                "ratio": round(self.ratio, 2),
                "image_bearing_deg": round(self.image_bearing_deg, 1),
                "confidence": self.confidence, "source": self.source,
                "notes": self.notes, "endpoints": self.endpoints}


def _shadow_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels plausibly in shadow.

    Shadow darkens a surface without changing what it is made of, so in LAB
    the lightness drops while the colour channels move comparatively little.

    Otsu over the non-sky pixels, rather than a fixed percentile. An earlier
    version took the 35th percentile of everything below the 85th, which
    collapses whenever the background is uniform: on a flat bright ground the
    "below 85th" set contains only the dark objects, and its 35th percentile
    then selects the subject instead of the shadow. Otsu finds the actual
    bimodal split between lit and shaded surfaces.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    # Sky and specular highlights would drag the split upward.
    sky_cut = float(np.percentile(lightness, 92))
    considered = lightness[lightness <= sky_cut]
    if considered.size < 100:
        return np.zeros(lightness.shape, np.uint8)

    thresh, _ = cv2.threshold(considered.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (lightness <= thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def _measure_from_person(mask: np.ndarray, box: tuple[int, int, int, int],
                         ) -> tuple[float, float, tuple[float, float, float, float]] | None:
    """Shadow length and image bearing for one detected person.

    Returns (length_px, bearing_deg) or None. The shadow is taken to be the
    dark region touching the subject's feet; its extent is measured as the
    farthest connected shadow pixel from that point, which is robust to the
    ragged edges real shadows have.
    """
    x, y, w, h = box
    foot = (x + w // 2, y + h)
    height_px = float(h)
    if height_px < 40:
        return None

    # Search a window scaled to the subject: a shadow longer than this would
    # mean a sun within a few degrees of the horizon.
    reach = int(height_px * MAX_RATIO)
    x0, x1 = max(0, foot[0] - reach), min(mask.shape[1], foot[0] + reach)
    # Shadows lie on the ground, so they cannot extend far above the feet.
    # Without this the search climbs the subject's own silhouette, which is
    # itself dark, and reports the person's height as their shadow.
    y0 = max(0, foot[1] - int(height_px * 0.15))
    y1 = min(mask.shape[0], foot[1] + reach)
    window = mask[y0:y1, x0:x1]
    if window.size == 0:
        return None

    count, labels = cv2.connectedComponents(window)
    fx, fy = foot[0] - x0, foot[1] - y0
    # The feet themselves may be a pixel outside the mask; search a small
    # neighbourhood for the component the shadow actually belongs to.
    label = 0
    for radius in (2, 5, 9, 14):
        ys = slice(max(0, fy - radius), min(window.shape[0], fy + radius + 1))
        xs = slice(max(0, fx - radius), min(window.shape[1], fx + radius + 1))
        patch = labels[ys, xs]
        nonzero = patch[patch > 0]
        if nonzero.size:
            label = int(np.bincount(nonzero).argmax())
            break
    if not label or count < 2:
        return None

    pts = np.column_stack(np.where(labels == label))      # (row, col)
    if pts.shape[0] < 30:
        return None
    d = np.hypot(pts[:, 0] - fy, pts[:, 1] - fx)
    far = int(np.argmax(d))
    length = float(d[far])
    if length < 5:
        return None

    dy = float(pts[far, 0] - fy)
    dx = float(pts[far, 1] - fx)
    # Clockwise from image-up, so it reads like a bearing once a heading is
    # supplied.
    bearing = (math.degrees(math.atan2(dx, -dy))) % 360.0
    tip_uv = (float(x0 + pts[far, 1]), float(y0 + pts[far, 0]))
    return length, bearing, (float(foot[0]), float(foot[1]),
                             tip_uv[0], tip_uv[1])


def measure(path: Path, person_boxes: list[tuple[int, int, int, int]] | None = None,
            ) -> ShadowMeasurement | None:
    """Estimate the sun's elevation from shadows cast by detected people."""
    if not person_boxes:
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None

    mask = _shadow_mask(bgr)
    samples: list[tuple[float, float, float]] = []      # ratio, bearing, height
    endpoints: list[tuple[float, float, float, float]] = []
    for box in person_boxes:
        got = _measure_from_person(mask, box)
        if got is None:
            continue
        length, bearing, ends = got
        ratio = length / float(box[3])
        if MIN_RATIO <= ratio <= MAX_RATIO:
            samples.append((ratio, bearing, float(box[3])))
            endpoints.append(ends)

    if not samples:
        return None

    ratios = np.array([s[0] for s in samples])
    bearings = np.array([s[1] for s in samples])
    ratio = float(np.median(ratios))
    # Circular median via the unit-vector mean; a plain mean would average
    # 350 and 10 degrees to 180.
    ang = np.radians(bearings)
    bearing = float(math.degrees(math.atan2(np.sin(ang).mean(),
                                            np.cos(ang).mean())) % 360.0)
    elevation = math.degrees(math.atan2(1.0, ratio))

    notes: list[str] = []
    confidence = "medium"
    if len(samples) == 1:
        notes.append("Only one shadow could be measured; a single subject may "
                     "be leaning, seated, or standing on a slope.")
        confidence = "low"
    spread = float(np.std(ratios)) if len(ratios) > 1 else 0.0
    if spread > 0.5 * max(ratio, 1e-6):
        notes.append(f"Shadow ratios disagree markedly across {len(samples)} "
                     "subjects, so the ground is probably not flat or some "
                     "subjects are not upright.")
        confidence = "low"

    # Toward-or-away-from camera shadows are compressed by perspective.
    vertical_offset = min(abs(bearing - 0.0), abs(bearing - 180.0),
                          abs(bearing - 360.0))
    if vertical_offset < FORESHORTEN_GUARD_DEG:
        notes.append("Shadows run toward or away from the camera, where "
                     "perspective shortens them. The true sun is probably "
                     "LOWER than this estimate.")
        confidence = "low"

    return ShadowMeasurement(
        elevation_deg=elevation, ratio=ratio, image_bearing_deg=bearing,
        confidence=confidence, source=f"{len(samples)} human shadow(s)",
        notes=notes, endpoints=endpoints)
