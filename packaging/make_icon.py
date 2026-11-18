"""Generate packaging/geoloc.icns from scratch.

Kept as code rather than a committed binary so the icon is reviewable and
reproducible. Draws at 1024px and lets iconutil downsample.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
S = 1024


def rounded_mask(size: int, radius_ratio: float = 0.2237) -> Image.Image:
    """macOS squircle-ish mask. 0.2237 approximates the Big Sur corner."""
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * radius_ratio), fill=255)
    return m


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Vertical gradient background: deep navy -> slate.
    top, bottom = (18, 26, 46), (32, 52, 84)
    for y in range(S):
        t = y / S
        d.line([(0, y), (S, y)],
               fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))

    inset = S * 0.13
    cx = cy = S / 2
    r = (S - 2 * inset) / 2

    # Graticule: latitude lines flatten toward the poles, meridians bow,
    # so the sphere reads as a globe rather than a flat disc.
    grid = (120, 190, 255, 70)
    for frac in (-0.66, -0.33, 0.0, 0.33, 0.66):
        y = cy + r * frac
        half = r * math.cos(math.asin(max(-1.0, min(1.0, frac))))
        w = 7 if frac == 0.0 else 4
        d.line([(cx - half, y), (cx + half, y)], fill=grid, width=w)
    for frac in (-0.75, -0.4, 0.0, 0.4, 0.75):
        pts = []
        for i in range(65):
            a = -math.pi / 2 + math.pi * i / 64
            pts.append((cx + r * frac * math.cos(a), cy + r * math.sin(a)))
        d.line(pts, fill=grid, width=7 if frac == 0.0 else 4, joint="curve")

    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(150, 205, 255, 165), width=9)

    # Map pin, offset up-left so it reads as sitting on the globe.
    px, py = cx - S * 0.055, cy - S * 0.085
    pr = S * 0.132
    accent = (255, 106, 74)
    d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=accent)
    tip = (px, py + pr * 2.55)
    spread = pr * 0.80
    d.polygon([(px - spread, py + pr * 0.62), (px + spread, py + pr * 0.62), tip],
              fill=accent)
    hr = pr * 0.40
    d.ellipse([px - hr, py - hr, px + hr, py + hr], fill=(20, 28, 48))

    img.putalpha(rounded_mask(S))
    return img


def main() -> int:
    icon = draw_icon()
    iconset = HERE / "geoloc.iconset"
    if iconset.exists():
        for f in iconset.iterdir():
            f.unlink()
    iconset.mkdir(exist_ok=True)

    for size in (16, 32, 64, 128, 256, 512):
        icon.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(
            iconset / f"icon_{size}x{size}@2x.png")
    icon.save(iconset / "icon_512x512@2x.png")

    out = HERE / "geoloc.icns"
    res = subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
                         capture_output=True)
    if res.returncode != 0:
        print(res.stderr.decode(), file=sys.stderr)
        return 1
    icon.resize((512, 512), Image.LANCZOS).save(HERE / "geoloc-icon.png")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
