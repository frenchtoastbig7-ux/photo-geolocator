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
    CCTLD_COUNTRY,
    PHONE_PREFIX_COUNTRY,
    SCRIPT_COUNTRIES,
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
PHONE_RE = re.compile(r"(?:\+|00)\s?(\d{1,4})[\s\-.\d()]{5,}")
# Vehicle registration patterns that are distinctive enough to be worth flagging
PLATE_HINTS = {
    "EU-band plate (blue strip)": re.compile(r"\b[A-Z]{1,3}[\s-]?\d{1,4}[\s-]?[A-Z]{1,3}\b"),
    "UK current format": re.compile(r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b"),
    "US-style plate": re.compile(r"\b[A-Z0-9]{5,7}\b"),
}


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


def _vision_ocr(path: Path) -> list[dict]:
    """Multi-pass Apple Vision OCR covering every supported writing system."""
    merged: list[dict] = []
    for pass_name, languages in OCR_PASSES.items():
        try:
            blocks = _vision_pass(path, languages)
        except Exception:
            continue
        for b in blocks:
            if not b["text"].strip() or not _accepts(b["text"], pass_name):
                continue
            b["script_pass"] = pass_name
            # Two passes reading the same region keep the more confident read.
            dup = next((m for m in merged
                        if _iou(m["bbox"], b["bbox"]) > 0.5), None)
            if dup is None:
                merged.append(b)
            elif b["confidence"] > dup["confidence"]:
                merged[merged.index(dup)] = b
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
    blocks, engine = run_ocr(path)
    if engine == "none":
        return [Evidence(
            id="ocr.unavailable", analyzer="text",
            title="No OCR engine available",
            detail="Install pyobjc (macOS) or easyocr to enable text extraction.",
            confidence=Confidence.LOW, tags=["tooling"],
        )]

    strong = [b for b in blocks if b["confidence"] >= 0.3]
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

    # ---- diacritic fingerprint -----------------------------------------
    special = {c for c in full_text if c in "ąćęłńóśźżğışçöüåäøæðþõñčšžřůěŧđ"}
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
