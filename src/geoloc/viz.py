"""Posterior rendering.

Two consumers with different needs:

* the HTML report wants a standalone equirectangular PNG that works with no
  map server and no network, and
* the Leaflet map wants projection-correct geometry, which an equirectangular
  raster is not once Leaflet reprojects to Web Mercator. It gets a list of
  lat/lon rectangles instead, so the overlay lines up exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Perceptually-ordered ramp (dark blue -> cyan -> yellow -> red). Chosen so
# that ordering survives greyscale printing, which report annexes often get.
_RAMP = np.array([
    (12, 18, 48), (26, 60, 120), (24, 120, 156), (54, 178, 140),
    (150, 210, 90), (238, 214, 70), (244, 148, 44), (222, 68, 40),
], dtype=np.float64)


def _colormap(t: np.ndarray) -> np.ndarray:
    """Map t in [0,1] to RGB via linear interpolation over the ramp."""
    t = np.clip(t, 0.0, 1.0) * (len(_RAMP) - 1)
    lo = np.floor(t).astype(int)
    hi = np.clip(lo + 1, 0, len(_RAMP) - 1)
    f = (t - lo)[..., None]
    return _RAMP[lo] * (1 - f) + _RAMP[hi] * f


def render_posterior_png(post: np.ndarray, grid, dest: Path,
                         upscale: int = 3) -> Path:
    """Equirectangular posterior heatmap with a faint land backdrop."""
    # Log stretch: posteriors are extremely peaked and a linear ramp shows
    # one bright pixel on black, which tells the analyst nothing about the
    # shape of the uncertainty.
    p = post / (post.max() or 1.0)
    t = np.log10(np.maximum(p, 1e-8))
    t = (t - t.min()) / (float(np.ptp(t)) or 1.0)

    rgb = _colormap(t)

    # Faint coastline-ish backdrop from the habitation distance field.
    land = grid.dist_km < 250.0
    edge = land ^ np.roll(land, 1, axis=1)
    rgb[edge] = np.clip(rgb[edge] * 0.4 + 140, 0, 255)

    img = Image.fromarray(rgb.astype(np.uint8)[::-1])  # flip: row 0 is -90 lat
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "PNG")
    return dest


def posterior_cells(post: np.ndarray, grid, *, top_n: int = 1200,
                    min_share: float = 1e-5) -> list[dict]:
    """Top posterior cells as lat/lon rectangles for a Leaflet overlay."""
    flat = post.ravel()
    n = min(top_n, flat.size)
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    peak = float(flat.max()) or 1.0
    half = grid.step / 2.0
    out = []
    for i in idx:
        v = float(flat[i])
        if v <= 0 or v / peak < min_share:
            break
        r, c = divmod(int(i), grid.shape[1])
        lat = float(grid.lat2d[r, c])
        lon = float(grid.lon2d[r, c])
        out.append({
            "bounds": [[lat - half, lon - half], [lat + half, lon + half]],
            "p": v, "rel": v / peak,
        })
    return out


def render_thumbnail(src: Path, dest: Path, max_side: int = 900) -> Path:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=88)
    return dest
