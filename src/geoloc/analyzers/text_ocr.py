"""Text extraction and linguistic geolocation.

Legible text is usually the single most decisive cue in a street-level
photograph: a shop name, a road sign, a phone number on a van, a ccTLD on a
hoarding. This analyzer pulls text out, then mines it for hard identifiers.

OCR runs through Apple's Vision framework where available -- it is on-device,
covers a wide range of scripts, needs no model download and never touches
the network, which suits the tool's threat model exactly.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from ..data.reference import (
    AIRCRAFT_PREFIX_COUNTRY,
    CCTLD_COUNTRY,
    DE_PLATE_CITY,
    PHONE_PREFIX_COUNTRY,
    POSTAL_PATTERNS,
    ROAD_PATTERNS,
    SCRIPT_COUNTRIES,
    UIC_COUNTRY,
    UIC_TYPE_PREFIXES,
)
from ..models import Confidence, ConstraintKind, Evidence, GeoConstraint

# Unicode block prefix (from unicodedata.name) -> script label
_SCRIPT_PREFIXES = (
    ("CJK", "Han"), ("HIRAGANA", "Hiragana"), ("KATAKANA", "Katakana"),
    ("HANGUL", "Hangul"), ("CYRILLIC", "Cyrillic"), ("GREEK", "Greek"),
    ("ARABIC", "Arabic"), ("HEBREW", "Hebrew"), ("THAI", "Thai"),
    ("LAO", "Lao"), ("KHMER", "Khmer"), ("MYANMAR", "Myanmar"),
    ("DEVANAGARI", "Devanagari"), ("BENGALI", "Bengali"), ("TAMIL", "Tamil"),
    ("TELUGU", "Telugu"), ("KANNADA", "Kannada"), ("MALAYALAM", "Malayalam"),
    ("GUJARATI", "Gujarati"), ("GURMUKHI", "Gurmukhi"), ("SINHALA", "Sinhala"),
    ("GEORGIAN", "Georgian"), ("ARMENIAN", "Armenian"), ("ETHIOPIC", "Ethiopic"),
    ("TIBETAN", "Tibetan"), ("MONGOLIAN", "Mongolian"), ("LATIN", "Latin"),
)

TLD_RE = re.compile(r"\b[\w-]{2,}(\.[a-z]{2,6})+\b", re.IGNORECASE)
# UIC vehicle numbers: 12 digits, conventionally written "93 87 0029 001-2".
# Grab any run of digits with optional separators; validation happens in
# `parse_uic`, which is where the real work is.
# Deliberately excludes \n: `\s` glued a neighbouring block's stray digit
# onto the number ("4\n93 87 0029...") and pushed it past the length check,
# so the number was found and then silently discarded.
UIC_RE = re.compile(r"\b\d[\d .·-]{6,20}\d\b")
AIRCRAFT_RE = re.compile(r"\b([A-Z]{1,2}\d?-[A-Z]{3,5}|N\d{1,5}[A-Z]{0,2})\b")
PHONE_RE = re.compile(r"(?:\+|00)\s?(\d{1,4})[\s\-.\d()]{5,}")
# Vehicle registration patterns that are distinctive enough to be worth flagging
PLATE_HINTS = {
    "EU-band plate (blue strip)": re.compile(r"\b[A-Z]{1,3}[\s-]?\d{1,4}[\s-]?[A-Z]{1,3}\b"),
    "UK current format": re.compile(r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b"),
    "US-style plate": re.compile(r"\b[A-Z0-9]{5,7}\b"),
}


def uic_check_digit(first_eleven: str) -> int:
    """UIC/Luhn self-check digit for the first 11 digits of a vehicle number.

    Digits are weighted 2,1,2,1... from the left; each product is reduced by
    summing its own digits; the check digit is what brings the total up to a
    multiple of ten.
    """
    total = 0
    for i, ch in enumerate(first_eleven):
        product = int(ch) * (2 if i % 2 == 0 else 1)
        total += product // 10 + product % 10
    return (10 - total % 10) % 10


def parse_uic(raw: str) -> str | None:
    """Country for a UIC vehicle number, or None if it is not one.

    Validation matters more than detection here: an unchecked run of digits
    would inject a HIGH-confidence country constraint from a price tag or a
    phone number. A full 12-digit number is accepted only when its check
    digit verifies. A partial number -- the usual case, since OCR rarely
    recovers a whole train flank -- is accepted only if the country code is
    real and the leading pair is a plausible vehicle type.
    """
    digits = re.sub(r"\D", "", raw)
    if not 8 <= len(digits) <= 12:
        return None

    country = UIC_COUNTRY.get(digits[2:4])
    if not country:
        return None

    if len(digits) == 12:
        return country if uic_check_digit(digits[:11]) == int(digits[11]) else None

    # Without the check digit, insist the type code is a real vehicle class.
    return country if digits[:2] in UIC_TYPE_PREFIXES else None


# A plate is written "B-XY 1234" / "M AB 123": district prefix, then a
# letter group, then digits. Anchored to avoid swallowing ordinary words.
DE_PLATE_RE = re.compile(r"\b([A-ZÄÖÜ]{1,3})[\s-]?([A-Z]{1,2})[\s-]?(\d{1,4})\b")


def postal_candidates(text: str) -> dict[str, list[str]]:
    """Postcode-shaped tokens, grouped by the countries whose form they fit.

    Shape is frequently ambiguous -- five bare digits fit six countries at
    once -- so this reports every country a token could belong to and leaves
    the choice to fusion. Only the distinctively-shaped formats (a British
    outward code, a Dutch "1012 AB", a Polish "00-001") identify a country on
    their own.
    """
    hits: dict[str, list[str]] = {}
    for iso2, (pattern, _distinctive) in POSTAL_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            token = m.group(0).strip().upper()
            hits.setdefault(iso2, [])
            if token not in hits[iso2]:
                hits[iso2].append(token)
    return hits


def road_candidates(text: str) -> dict[str, list[str]]:
    """Road numbers matching national numbering conventions."""
    hits: dict[str, list[str]] = {}
    for iso2, patterns in ROAD_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                token = m.group(0).strip()
                hits.setdefault(iso2, [])
                if token not in hits[iso2]:
                    hits[iso2].append(token)
    return hits


def classify_scripts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for prefix, label in _SCRIPT_PREFIXES:
            if name.startswith(prefix):
                counts[label] += 1
                break
    return counts


# Vision treats `recognitionLanguages` as an ordered *priority* list, not a
# set: whichever language leads the list picks the recognition model, and
# everything in a different writing system is then missed entirely. Passing
# the full supported-language list returns nothing at all. So OCR runs once
# per script family, and the results are merged.
#
# Verified on Vision revision 3: ["ja-JP"] reads Japanese signage at
# confidence 1.0, while ["en-US", "ja-JP"] returns zero blocks.
OCR_PASSES: dict[str, list[str]] = {
    "Latin": ["en-US", "fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR", "nl-NL",
              "pl-PL", "cs-CZ", "tr-TR", "ro-RO", "sv-SE", "da-DK", "nb-NO",
              "ms-MY", "id-ID", "vi-VT"],
    "Cyrillic": ["ru-RU", "uk-UA"],
    "Han": ["zh-Hans", "zh-Hant"],
    "Hiragana": ["ja-JP"],
    "Hangul": ["ko-KR"],
    "Arabic": ["ar-SA"],
    "Thai": ["th-TH"],
}

# A pass run against the wrong script does not fail quietly -- it invents
# plausible-looking transliterations (the Chinese pass renders Cyrillic
# "Улица Ленина" as "ynuua JeHHa"). Each pass's output is therefore kept only
# if it actually contains characters of the script that pass was for.
_PASS_ACCEPTS: dict[str, set[str]] = {
    "Latin": {"Latin"},
    "Cyrillic": {"Cyrillic"},
    "Han": {"Han"},
    "Hiragana": {"Hiragana", "Katakana", "Han"},
    "Hangul": {"Hangul"},
    "Arabic": {"Arabic"},
    "Thai": {"Thai"},
}


def _vision_pass(path: Path, languages: list[str]) -> list[dict]:
    """One Vision OCR request with a fixed language priority list."""
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(path.resolve()))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    request.setRecognitionLanguages_(languages)

    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")

    out = []
    for obs in (request.results() or []):
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        box = obs.boundingBox()
        out.append({
            "text": str(cands[0].string()),
            "confidence": float(cands[0].confidence()),
            # Vision's origin is bottom-left, normalised 0-1
            "bbox": [float(box.origin.x), float(box.origin.y),
                     float(box.size.width), float(box.size.height)],
        })
    return out


def _accepts(text: str, pass_name: str) -> bool:
    """True if `text` genuinely belongs to the script this pass targets."""
    counts = classify_scripts(text)
    wanted = _PASS_ACCEPTS[pass_name]
    if any(counts.get(s, 0) for s in wanted):
        return True
    # Digits and punctuation carry no script; keep them from the Latin pass
    # only, so a bare street number is not dropped.
    return pass_name == "Latin" and not counts and any(c.isdigit() for c in text)


def _iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_block(merged: list[dict], b: dict) -> None:
    """Insert a block, keeping the more confident of two overlapping reads."""
    dup = next((m for m in merged if _iou(m["bbox"], b["bbox"]) > 0.4), None)
    if dup is None:
        merged.append(b)
    elif b["confidence"] > dup["confidence"]:
        merged[merged.index(dup)] = b


def _run_passes(path: Path, scripts: list[str]) -> list[dict]:
    """Run one Vision pass per named script family over a single image."""
    out: list[dict] = []
    for name in scripts:
        try:
            blocks = _vision_pass(path, OCR_PASSES[name])
        except Exception:
            continue
        for b in blocks:
            if b["text"].strip() and _accepts(b["text"], name):
                b["script_pass"] = name
                out.append(b)
    return out


# A shopfront or street sign occupies a small fraction of a wide photograph,
# and Vision simply does not return it: the signage in a 1652x1266 campus
# photo reads as nothing full-frame, yet reads correctly once the region is
# cropped and enlarged. Since legible text is usually the single most
# decisive geolocation cue, missing all of it is the worst failure this tool
# can have -- so large images are also scanned as overlapping tiles.
TILE_MIN_PIXELS = 900 * 700
TILE_GRID = (3, 3)
TILE_OVERLAP = 0.25
TILE_UPSCALE = 2


def _tiled_ocr(path: Path, scripts: list[str]) -> list[dict]:
    """OCR overlapping enlarged tiles, mapped back to whole-image coordinates."""
    import tempfile

    from PIL import Image

    merged: list[dict] = []
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        cols, rows = TILE_GRID
        pad_x = int(width / cols * TILE_OVERLAP)
        pad_y = int(height / rows * TILE_OVERLAP)

        with tempfile.TemporaryDirectory(prefix="geoloc-ocr-") as tmp:
            tmp_dir = Path(tmp)
            for row in range(rows):
                for col in range(cols):
                    x0 = max(0, int(col * width / cols) - pad_x)
                    x1 = min(width, int((col + 1) * width / cols) + pad_x)
                    y0 = max(0, int(row * height / rows) - pad_y)
                    y1 = min(height, int((row + 1) * height / rows) + pad_y)
                    if x1 - x0 < 40 or y1 - y0 < 40:
                        continue

                    tile = im.crop((x0, y0, x1, y1))
                    tile = tile.resize(
                        (tile.width * TILE_UPSCALE, tile.height * TILE_UPSCALE),
                        Image.LANCZOS)
                    tile_path = tmp_dir / f"tile_{col}_{row}.png"
                    tile.save(tile_path)

                    for b in _run_passes(tile_path, scripts):
                        # Vision returns normalised, bottom-left-origin boxes
                        # relative to the tile; re-express them against the
                        # whole image so dedup and reporting stay consistent.
                        tx, ty, tw, th = b["bbox"]
                        tile_w = x1 - x0
                        tile_h = y1 - y0
                        b["bbox"] = [
                            (x0 + tx * tile_w) / width,
                            (y0 / height) + ty * (tile_h / height),
                            tw * tile_w / width,
                            th * tile_h / height,
                        ]
                        b["tiled"] = True
                        _merge_block(merged, b)
    return merged


def _vision_ocr(path: Path) -> list[dict]:
    """Multi-pass Apple Vision OCR covering every supported writing system.

    Two stages: a whole-image pass per script family, then -- for images big
    enough for it to matter -- a tiled rescan that recovers small signage the
    whole-image pass cannot see.
    """
    merged: list[dict] = []
    for b in _run_passes(path, list(OCR_PASSES)):
        _merge_block(merged, b)

    try:
        from PIL import Image

        with Image.open(path) as im:
            big_enough = im.size[0] * im.size[1] >= TILE_MIN_PIXELS
    except Exception:
        return merged

    if not big_enough:
        return merged

    # Tiling multiplies the pass count by the number of tiles, so restrict it
    # to the scripts actually seen full-frame. When nothing was seen at all --
    # precisely the case this exists to fix -- fall back to the scripts that
    # cover most of the world's signage.
    seen = {b["script_pass"] for b in merged}
    scripts = sorted(seen) if seen else ["Latin", "Han", "Cyrillic", "Arabic"]

    for b in _tiled_ocr(path, scripts):
        _merge_block(merged, b)
    return merged


def run_ocr(path: Path) -> tuple[list[dict], str]:
    if sys.platform == "darwin":
        try:
            return _vision_ocr(path), "apple-vision"
        except Exception:
            pass
    try:
        import easyocr  # optional, heavier fallback
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        res = reader.readtext(str(path))
        return ([{"text": t, "confidence": float(c), "bbox": None}
                 for _, t, c in res], "easyocr")
    except Exception:
        return [], "none"


def analyze(path: Path) -> list[Evidence]:
    """Read the image and turn whatever text it holds into evidence."""
    blocks, engine = run_ocr(path)
    return evidence_from_blocks(blocks, engine)


def evidence_from_blocks(blocks: list[dict], engine: str = "apple-vision"
                         ) -> list[Evidence]:
    """Derive evidence from OCR output.

    Split from `analyze` so the interpretation of text -- postcodes, road
    numbers, plates, scripts, diacritics -- can be tested against literal
    strings instead of requiring a rendered image and a working OCR engine.
    Most of the defects in this module were in the interpretation, not the
    reading.
    """
    if engine == "none":
        return [Evidence(
            id="ocr.unavailable", analyzer="text",
            title="No OCR engine available",
            detail="Install pyobjc (macOS) or easyocr to enable text extraction.",
            confidence=Confidence.LOW, tags=["tooling"],
        )]

    strong = [b for b in blocks if b.get("confidence", 1.0) >= 0.3]
    full_text = "\n".join(b["text"] for b in strong)
    out: list[Evidence] = []

    if not strong:
        return [Evidence(
            id="ocr.empty", analyzer="text",
            title="No legible text detected",
            detail=f"{engine} found no text above the confidence threshold.",
            confidence=Confidence.MEDIUM, tags=["text"],
        )]

    out.append(Evidence(
        id="ocr.text", analyzer="text",
        title=f"Extracted text ({len(strong)} blocks)",
        detail=full_text[:2000],
        confidence=Confidence.HIGH, tags=["text"],
        raw={"engine": engine, "blocks": strong[:200]},
    ))

    # ---- script -> country prior ---------------------------------------
    scripts = classify_scripts(full_text)
    informative = {s: n for s, n in scripts.items()
                   if s != "Latin" and n >= 2 and SCRIPT_COUNTRIES.get(s)}
    if informative:
        top_script = max(informative, key=informative.get)
        countries = SCRIPT_COUNTRIES[top_script]
        out.append(Evidence(
            id="ocr.script", analyzer="text",
            title=f"Non-Latin script detected: {top_script}",
            detail=f"{informative[top_script]} {top_script} characters. "
                   f"Routine public use in: {', '.join(countries)}.",
            confidence=Confidence.HIGH, tags=["text", "script"],
            raw={"scripts": dict(scripts)},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(countries, 1.0),
                floor=0.03, confidence=Confidence.HIGH,
                note=f"{top_script} script",
            )],
        ))
    elif scripts:
        out.append(Evidence(
            id="ocr.script_latin", analyzer="text",
            title="Latin script only",
            detail="Latin script alone does not constrain location; check "
                   "diacritics and vocabulary instead.",
            confidence=Confidence.LOW, tags=["text", "script"],
            raw={"scripts": dict(scripts)},
        ))

    # ---- ccTLD ---------------------------------------------------------
    tld_hits: dict[str, str] = {}
    for m in TLD_RE.finditer(full_text):
        domain = m.group(0).lower()
        for tld, cc in CCTLD_COUNTRY.items():
            if domain.endswith(tld):
                tld_hits[domain] = cc
                break
    if tld_hits:
        ccs = sorted(set(tld_hits.values()))
        out.append(Evidence(
            id="ocr.cctld", analyzer="text",
            title=f"Country-code domain(s): {', '.join(ccs)}",
            detail="Found: " + "; ".join(f"{d} -> {c}" for d, c in tld_hits.items())
                   + ". A local ccTLD on signage is a strong locality indicator.",
            confidence=Confidence.HIGH, tags=["text", "domain"],
            raw={"domains": tld_hits},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(ccs, 1.0),
                floor=0.05, confidence=Confidence.HIGH, note="ccTLD on signage",
            )],
        ))

    # ---- international dialling prefix ---------------------------------
    phone_hits: dict[str, str] = {}
    for m in PHONE_RE.finditer(full_text):
        digits = m.group(1)
        for length in (4, 3, 2, 1):
            if len(digits) >= length and digits[:length] in PHONE_PREFIX_COUNTRY:
                phone_hits[m.group(0).strip()] = PHONE_PREFIX_COUNTRY[digits[:length]]
                break
    if phone_hits:
        ccs = sorted(set(phone_hits.values()))
        out.append(Evidence(
            id="ocr.phone", analyzer="text",
            title=f"International dialling prefix -> {', '.join(ccs)}",
            detail="Found: " + "; ".join(f"{p} -> {c}" for p, c in phone_hits.items()),
            confidence=Confidence.HIGH, tags=["text", "phone"],
            raw={"numbers": phone_hits},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(ccs, 1.0),
                floor=0.05, confidence=Confidence.HIGH, note="phone country code",
            )],
        ))

    # ---- rolling-stock numbers ------------------------------------------
    uic_hits: dict[str, str] = {}
    # Scanned per OCR block: a vehicle number is a single painted string, and
    # concatenating blocks invents digit runs that never appeared on anything.
    for block in strong:
        for m in UIC_RE.finditer(str(block.get("text", ""))):
            parsed = parse_uic(m.group(0))
            if parsed:
                uic_hits[m.group(0).strip()] = parsed
    if uic_hits:
        ccs = sorted(set(uic_hits.values()))
        out.append(Evidence(
            id="ocr.uic", analyzer="text",
            title=f"Rail vehicle number -> {', '.join(ccs)}",
            detail="Found: " + "; ".join(f"{n} -> {c}" for n, c in uic_hits.items())
                   + ". Digits 3-4 of a UIC number are the keeper country, so "
                     "this is a registration fact rather than an inference. "
                     "Note it identifies where the vehicle is registered, not "
                     "necessarily where the photo was taken -- international "
                     "services run foreign stock.",
            confidence=Confidence.HIGH, tags=["text", "rail", "vehicle"],
            raw={"numbers": uic_hits},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(ccs, 1.0),
                floor=0.08, confidence=Confidence.HIGH,
                note="UIC rolling-stock country code",
            )],
        ))

    # ---- aircraft registrations ------------------------------------------
    air_hits: dict[str, str] = {}
    for m in AIRCRAFT_RE.finditer(full_text.upper()):
        reg = m.group(0)
        for prefix, iso2 in sorted(AIRCRAFT_PREFIX_COUNTRY.items(),
                                   key=lambda kv: -len(kv[0])):
            if reg.startswith(prefix):
                air_hits[reg] = iso2
                break
    if air_hits:
        ccs = sorted(set(air_hits.values()))
        out.append(Evidence(
            id="ocr.aircraft", analyzer="text",
            title=f"Aircraft registration -> {', '.join(ccs)}",
            detail="Found: " + "; ".join(f"{n} -> {c}" for n, c in air_hits.items())
                   + ". Registration prefixes are allocated by ICAO, so the "
                     "country of registry is certain -- though an aircraft "
                     "may of course be photographed anywhere.",
            confidence=Confidence.MEDIUM, tags=["text", "aviation", "vehicle"],
            raw={"registrations": air_hits},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(ccs, 1.0),
                floor=0.25, confidence=Confidence.LOW,
                note="aircraft country of registry",
            )],
        ))

    # ---- German registration plates --------------------------------------
    plate_hits: dict[str, tuple[str, float, float]] = {}
    for m in DE_PLATE_RE.finditer(full_text.upper()):
        city = DE_PLATE_CITY.get(m.group(1))
        if city:
            plate_hits[m.group(0)] = city
    if plate_hits:
        cities = {v[0]: (v[1], v[2]) for v in plate_hits.values()}
        out.append(Evidence(
            id="ocr.plate_de", analyzer="text",
            title=f"German plate district -> {', '.join(sorted(cities))}",
            detail="Read: " + "; ".join(f"{k} -> {v[0]}"
                                        for k, v in plate_hits.items())
                   + ". The prefix is the registration district, so it names "
                     "where the vehicle is registered -- which is usually, "
                     "but not always, where it is photographed.",
            confidence=Confidence.MEDIUM, tags=["text", "vehicle", "plate"],
            raw={"plates": {k: v[0] for k, v in plate_hits.items()}},
            constraints=[GeoConstraint(
                kind=ConstraintKind.SITES,
                sites=[(lat, lon) for lat, lon in cities.values()],
                site_radius_km=35.0,
                # A per-cell floor is applied across ~259k grid cells, so
                # anything much above 1e-4 lets the world outside the sites
                # outweigh them by sheer count. At 0.05 a legible Berlin
                # plate moved the posterior not at all: Germany did not even
                # reach the top six countries.
                floor=1e-4,
                confidence=Confidence.MEDIUM,
                note="German registration district")],
        ))

    # ---- postal codes ----------------------------------------------------
    # Strip anything already claimed by a plate or a road number first: the
    # "1234" in "B-XY 1234" is a registration, and letting it register as a
    # four-digit postcode added twelve spurious countries to the shortlist.
    postal_text = full_text
    for claimed in (list(plate_hits) if plate_hits else []) + [
            t for toks in road_candidates(full_text).values() for t in toks]:
        postal_text = postal_text.replace(claimed, " ")
    postal = postal_candidates(postal_text)
    if postal:
        distinctive = {c: toks for c, toks in postal.items()
                       if POSTAL_PATTERNS[c][1]}
        # A distinctively-shaped code names its country; an ambiguous one only
        # says "one of these", so it must not out-vote real evidence.
        chosen = distinctive or postal
        confidence = Confidence.HIGH if distinctive else Confidence.LOW
        sample = sorted({t for toks in chosen.values() for t in toks})[:4]
        out.append(Evidence(
            id="ocr.postal", analyzer="text",
            title=f"Postcode-shaped text -> {', '.join(sorted(chosen))}",
            detail=(f"Found {', '.join(repr(s) for s in sample)}. "
                    + ("The format identifies the country on its own."
                       if distinctive else
                       "This shape is shared by several countries, so it "
                       "narrows nothing by itself -- but once other evidence "
                       "settles the country it can be resolved to a town.")),
            confidence=confidence, tags=["text", "postal"],
            raw={"by_country": chosen, "ambiguous": not distinctive},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(chosen, 1.0),
                floor=0.10 if distinctive else 0.55,
                confidence=confidence,
                note="postcode format")],
        ))

    # ---- road numbering --------------------------------------------------
    roads = road_candidates(full_text)
    if roads:
        sample = {c: t[:2] for c, t in list(roads.items())[:5]}
        out.append(Evidence(
            id="ocr.road", analyzer="text",
            title=f"Road numbering consistent with {', '.join(sorted(roads))}",
            detail="Matched: " + "; ".join(f"{c}: {', '.join(t)}"
                                           for c, t in sample.items())
                   + ". Numbering conventions overlap heavily between "
                     "countries, so this corroborates rather than decides.",
            confidence=Confidence.LOW, tags=["text", "road"],
            raw={"by_country": roads},
            constraints=[GeoConstraint(
                kind=ConstraintKind.COUNTRIES,
                country_scores=dict.fromkeys(roads, 1.0),
                floor=0.55, confidence=Confidence.LOW,
                note="road numbering convention")],
        ))

    # ---- diacritic fingerprint -----------------------------------------
    # Lower-cased first: signage is overwhelmingly upper case, so a
    # case-sensitive membership test missed "BÄCKEREI" entirely.
    special = {c for c in full_text.lower() if c in "ąćęłńóśźżğışçöüåäøæðþõñčšžřůěŧđ"}
    if special:
        out.append(Evidence(
            id="ocr.diacritics", analyzer="text",
            title="Distinctive diacritics present",
            detail=f"Characters {''.join(sorted(special))} narrow the language "
                   "family considerably; cross-check against the language list.",
            confidence=Confidence.MEDIUM, tags=["text", "language"],
            raw={"characters": sorted(special)},
        ))

    return out
