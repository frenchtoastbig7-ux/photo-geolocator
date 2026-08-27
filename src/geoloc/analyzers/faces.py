"""Face presence detection and export redaction.

This is a guardrail, not an analytical capability. The tool geolocates
*scenes*; it deliberately performs no facial recognition, no face matching
and no identity inference. What it does:

* counts faces so the analyst is warned that the image contains people, and
* blurs them in any exported report imagery,

so that a scene-geolocation product does not quietly become a
person-tracking product. Detection is bounding-box only; no face embedding
is ever computed or stored.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

from ..models import FaceFinding


def vision_box_to_pixels(x: float, y: float, w: float, h: float,
                         img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert a Vision bounding box to an OpenCV-style pixel box.

    Vision reports normalised coordinates with a **bottom-left** origin;
    OpenCV and PIL use pixels with a **top-left** origin. Getting this
    inversion wrong redacts a mirrored region of the image and leaves the
    actual face visible, so it is isolated here and unit tested.
    """
    px = round(x * img_w)
    pw = round(w * img_w)
    ph = round(h * img_h)
    py = round((1.0 - y - h) * img_h)
    return px, py, pw, ph


def _vision_detect(path: Path, w_img: int, h_img: int
                   ) -> list[tuple[int, int, int, int]] | None:
    """Apple Vision face rectangles. On-device, no model download.

    Vision returns normalised boxes with a bottom-left origin; convert to
    pixel boxes with a top-left origin to match OpenCV convention.
    """
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return None

    url = NSURL.fileURLWithPath_(str(path.resolve()))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return None

    boxes = []
    for obs in (request.results() or []):
        b = obs.boundingBox()
        boxes.append(vision_box_to_pixels(
            b.origin.x, b.origin.y, b.size.width, b.size.height, w_img, h_img))
    return boxes


def _yunet_detect(bgr: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    """OpenCV YuNet DNN detector, if its weights are cached locally."""
    from ..config import SETTINGS

    model_path = SETTINGS.model_cache / "face_detection_yunet_2023mar.onnx"
    if not model_path.exists():
        return None
    try:
        h, w = bgr.shape[:2]
        det = cv2.FaceDetectorYN.create(str(model_path), "", (w, h),
                                        score_threshold=0.6)
        _, faces = det.detect(bgr)
    except cv2.error:
        return None
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def _merge_boxes(boxes: list[tuple[int, int, int, int]], iou_thresh: float = 0.3):
    """Cheap NMS so frontal+profile cascades don't double-count one face."""
    if not boxes:
        return []
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: -b[2] * b[3]):
        x1, y1, w1, h1 = box
        overlap = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x1 + w1, kx + kw) - max(x1, kx))
            iy = max(0, min(y1 + h1, ky + kh) - max(y1, ky))
            inter = ix * iy
            union = w1 * h1 + kw * kh - inter
            if union > 0 and inter / union > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(box)
    return kept


def detect_people(path: Path) -> int:
    """Count people present, whether or not their faces are visible.

    Face detection alone badly understates this. A station-platform photo
    with fifteen people walking away from the camera reports zero faces,
    which reads as "no privacy consideration here" when the image is in fact
    full of identifiable people -- identifiable by clothing, build, gait and
    companions, none of which needs a face. The warning should fire on
    people, not on faces.
    """
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return 0
    request_cls = getattr(Vision, "VNDetectHumanRectanglesRequest", None)
    if request_cls is None:
        return 0
    try:
        url = NSURL.fileURLWithPath_(str(path.resolve()))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        request = request_cls.alloc().init()
        with contextlib.suppress(Exception):
            request.setUpperBodyOnly_(False)
        ok, _err = handler.performRequests_error_([request], None)
        return len(request.results() or []) if ok else 0
    except Exception:
        return 0


def detect(path: Path) -> FaceFinding:
    """Locate faces so they can be flagged and redacted.

    Returns bounding boxes only. No embedding, descriptor or identity signal
    is computed at any point.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return FaceFinding(detector="unreadable")
    h_img, w_img = bgr.shape[:2]

    boxes = _vision_detect(path, w_img, h_img)
    detector = "apple-vision"
    if boxes is None:
        boxes = _yunet_detect(bgr)
        detector = "opencv-yunet"
    if boxes is None:
        return FaceFinding(detector="unavailable")

    merged = _merge_boxes(boxes)
    return FaceFinding(count=len(merged), boxes=merged, detector=detector,
                       people_count=detect_people(path))


def write_redacted(path: Path, finding: FaceFinding, dest: Path,
                   expand: float = 0.25) -> Path | None:
    """Write a copy of the image with detected faces blurred out.

    Boxes are expanded before blurring because Haar boxes crop tightly to the
    face and leave identifying hairline and jaw detail at the edges.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h_img, w_img = bgr.shape[:2]

    for x, y, w, h in finding.boxes:
        pad_x, pad_y = int(w * expand), int(h * expand)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        # Kernel scaled to region size so the blur is irreversible at any
        # resolution, then pixelated as a second pass.
        k = max(15, (min(x1 - x0, y1 - y0) // 3) | 1)
        roi = cv2.GaussianBlur(roi, (k, k), 0)
        small = cv2.resize(roi, (max(1, roi.shape[1] // 12), max(1, roi.shape[0] // 12)),
                           interpolation=cv2.INTER_LINEAR)
        bgr[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                        interpolation=cv2.INTER_NEAREST)

    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), bgr)
    return dest
