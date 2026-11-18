"""Zero-shot scene analysis with StreetCLIP.

StreetCLIP is a CLIP ViT-L/14 further pretrained on street-level imagery for
exactly this task, so a zero-shot country classification is meaningfully
better than vanilla CLIP. The same image embedding is reused across every
prompt axis, so the extra axes (biome, architecture, season, road markings)
cost one matrix multiply each rather than another forward pass.

Two cautions are wired into how the output is used downstream:

* Zero-shot country scores are systematically overconfident and biased toward
  countries heavily represented in street-level training data. They enter
  fusion at MEDIUM confidence with a generous floor, never as a verdict.
* CLIP cannot reliably read driving side, which is one of the strongest cues
  available. It is offered as a low-confidence guess and the UI invites the
  analyst to override it -- a human call here is worth more than the model's.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from ..config import SETTINGS
from ..data.reference import ARCHITECTURE_COUNTRIES, BIOME_LAT_BANDS, LEFT_DRIVING, PROMPT_AXES
from ..models import Confidence, ConstraintKind, Evidence, GeoConstraint

COUNTRY_PROMPT = "A Street View photo in {}."


@functools.lru_cache(maxsize=1)
def _country_list() -> list[tuple[str, str]]:
    """[(iso2, english_name)] for zero-shot country classification."""
    import geonamescache

    gc = geonamescache.GeonamesCache()
    return sorted(
        ((iso2, info["name"]) for iso2, info in gc.get_countries().items() if info.get("name")),
        key=lambda kv: kv[0],
    )


class SceneModelUnavailable(RuntimeError):
    """The scene model cannot be loaded, with a reason fit to show a user."""


@functools.lru_cache(maxsize=1)
def _load_model():
    from .. import modelmgr

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise SceneModelUnavailable(
            "PyTorch is not installed. Install the ML extra: "
            "uv pip install -e '.[ml]'"
        ) from exc

    snapshot = modelmgr.snapshot_path()
    if snapshot is None:
        raise SceneModelUnavailable(
            "The scene model is not installed. It is a one-time 1.7 GB "
            "download; every other analyzer works without it."
        )

    # Load from the resolved local snapshot with a pinned revision, so no
    # network call happens here even implicitly.
    device = SETTINGS.resolve_device()
    model = CLIPModel.from_pretrained(str(snapshot), local_files_only=True)
    model = model.to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True)
    return model, processor, device, torch



def _features(out):
    """Extract the projected embedding tensor from a CLIP feature call.

    transformers 4.x returns a bare tensor from `get_text_features` /
    `get_image_features`; 5.x returns a `BaseModelOutputWithPooling` whose
    `pooler_output` is the projected embedding. Verified equivalent: both
    reproduce `CLIPModel.forward`'s `logits_per_image` exactly once L2
    normalised.
    """
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and not hasattr(out, "norm"):
        return out.last_hidden_state
    return out


@functools.lru_cache(maxsize=64)
def _encode_prompts(prompts: tuple[str, ...]) -> np.ndarray:
    """Encode and L2-normalise a tuple of text prompts. Cached across images."""
    model, processor, device, torch = _load_model()
    with torch.no_grad():
        toks = processor(text=list(prompts), return_tensors="pt",
                         padding=True, truncation=True).to(device)
        feats = _features(model.get_text_features(**toks))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def _encode_image(path: Path) -> np.ndarray:
    from PIL import Image

    model, processor, device, torch = _load_model()
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        feats = _features(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def _softmax(x: np.ndarray, temperature: float = 100.0) -> np.ndarray:
    z = x * temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _rank(image_vec: np.ndarray, labels: list[str], template: str = "{}"
          ) -> list[tuple[str, float]]:
    text = _encode_prompts(tuple(template.format(lbl) for lbl in labels))
    probs = _softmax(text @ image_vec)
    return sorted(zip(labels, probs.tolist(), strict=True), key=lambda kv: -kv[1])


def analyze(path: Path, *, heatmaps: dict[str, np.ndarray] | None = None,
            grid=None) -> list[Evidence]:
    try:
        image_vec = _encode_image(path)
    except SceneModelUnavailable as exc:
        return [Evidence(
            id="clip.unavailable", analyzer="scene",
            title="Scene model not installed", detail=str(exc),
            confidence=Confidence.LOW, tags=["tooling"],
        )]
    except RuntimeError as exc:
        return [Evidence(
            id="clip.unavailable", analyzer="scene",
            title="Scene model unavailable", detail=str(exc),
            confidence=Confidence.LOW, tags=["tooling"],
        )]
    except OSError as exc:
        return [Evidence(
            id="clip.download_blocked", analyzer="scene",
            title="Scene model weights not available locally",
            detail=(f"Could not load '{SETTINGS.clip_model}' ({exc}). In offline "
                    "mode the weights must already be in the local HuggingFace "
                    "cache. Pre-fetch once with networking enabled, then the "
                    "tool runs fully air-gapped."),
            confidence=Confidence.LOW, tags=["tooling"],
        )]

    out: list[Evidence] = []

    # ---- country zero-shot (StreetCLIP's designed task) ------------------
    countries = _country_list()
    ranked = _rank(image_vec, [name for _, name in countries], COUNTRY_PROMPT)
    name_to_iso = {name: iso for iso, name in countries}
    top = ranked[:12]
    scores = {name_to_iso[n]: p for n, p in top if n in name_to_iso}
    out.append(Evidence(
        id="clip.country", analyzer="scene",
        title="Zero-shot country estimate",
        detail=" | ".join(f"{n} {p*100:.1f}%" for n, p in top[:8])
               + "\n\nStreetCLIP zero-shot output. Systematically "
                 "overconfident and biased toward well-represented countries; "
                 "weighted accordingly in fusion.",
        confidence=Confidence.MEDIUM, tags=["ml", "country"],
        raw={"ranked": [(n, float(p)) for n, p in ranked[:40]]},
        constraints=[GeoConstraint(
            kind=ConstraintKind.COUNTRIES, country_scores=scores,
            floor=0.10, confidence=Confidence.MEDIUM,
            note="StreetCLIP zero-shot country distribution",
        )],
    ))

    # ---- biome -> latitude band -----------------------------------------
    biomes = _rank(image_vec, PROMPT_AXES["biome"], "A photo of {} terrain.")
    b_label, b_prob = biomes[0]
    lo, hi = BIOME_LAT_BANDS[b_label]
    out.append(Evidence(
        id="clip.biome", analyzer="scene",
        title=f"Biome: {b_label}",
        detail=f"{b_prob*100:.0f}% confidence. Typical absolute-latitude band "
               f"{lo:g}-{hi:g} deg (either hemisphere). Runners-up: "
               + ", ".join(f"{n} {p*100:.0f}%" for n, p in biomes[1:4]),
        confidence=Confidence.MEDIUM if b_prob > 0.4 else Confidence.LOW,
        tags=["ml", "biome"],
        raw={"ranked": [(n, float(p)) for n, p in biomes]},
        constraints=([GeoConstraint(
            kind=ConstraintKind.LAT_BAND, lat_min=lo, lat_max=hi,
            softness_deg=8.0, floor=0.15,
            confidence=Confidence.LOW,
            note=f"northern band for {b_label}",
        ), GeoConstraint(
            kind=ConstraintKind.LAT_BAND, lat_min=-hi, lat_max=-lo,
            softness_deg=8.0, floor=0.15,
            confidence=Confidence.LOW,
            note=f"southern band for {b_label}",
        )] if b_label != "alpine" and b_prob > 0.35 else []),
    ))

    # ---- architecture -> country prior ----------------------------------
    arch = _rank(image_vec, PROMPT_AXES["architecture"], "A photo of {}.")
    a_label, a_prob = arch[0]
    arch_countries = ARCHITECTURE_COUNTRIES.get(a_label, [])
    out.append(Evidence(
        id="clip.architecture", analyzer="scene",
        title=f"Built environment: {a_label}",
        detail=f"{a_prob*100:.0f}% confidence. Associated with: "
               f"{', '.join(arch_countries) or 'n/a'}. Runners-up: "
               + ", ".join(f"{n} {p*100:.0f}%" for n, p in arch[1:3]),
        confidence=Confidence.LOW, tags=["ml", "architecture"],
        raw={"ranked": [(n, float(p)) for n, p in arch]},
        constraints=([GeoConstraint(
            kind=ConstraintKind.COUNTRIES,
            country_scores=dict.fromkeys(arch_countries, 1.0),
            floor=0.25, confidence=Confidence.SPECULATIVE,
            note=f"architecture style: {a_label}",
        )] if arch_countries and a_prob > 0.3 else []),
    ))

    # ---- descriptive axes (no constraint, analyst context) ---------------
    for axis, template, title in (
        ("setting", "A photo of a {}.", "Setting"),
        ("season", "A photo showing {}.", "Season"),
        ("utility", "A photo showing {}.", "Utility infrastructure"),
        ("road_markings", "A road with {}.", "Road markings"),
    ):
        ranked_axis = _rank(image_vec, PROMPT_AXES[axis], template)
        label, prob = ranked_axis[0]
        out.append(Evidence(
            id=f"clip.{axis}", analyzer="scene",
            title=f"{title}: {label}",
            detail=f"{prob*100:.0f}% confidence. Alternatives: "
                   + ", ".join(f"{n} {p*100:.0f}%" for n, p in ranked_axis[1:3]),
            confidence=Confidence.LOW, tags=["ml", axis],
            raw={"ranked": [(n, float(p)) for n, p in ranked_axis]},
        ))

    # ---- driving side (offered, not asserted) ----------------------------
    side = _rank(image_vec,
                 ["traffic driving on the left hand side of the road",
                  "traffic driving on the right hand side of the road"],
                 "{}")
    s_label, s_prob = side[0]
    is_left = "left" in s_label
    out.append(Evidence(
        id="clip.driving_side", analyzer="scene",
        title=f"Driving side (model guess): {'LEFT' if is_left else 'RIGHT'}",
        detail=f"{s_prob*100:.0f}% confidence -- LOW RELIABILITY. CLIP reads "
               "this poorly. If you can determine driving side yourself from "
               "vehicles, signage or road markings, override it: it is one of "
               "the strongest single cues available, splitting the world "
               f"{len(LEFT_DRIVING)}/{195 - len(LEFT_DRIVING)} by country.",
        confidence=Confidence.SPECULATIVE, tags=["ml", "driving"],
        raw={"left_probability": float(side[0][1] if is_left else side[1][1])},
    ))

    return out


def driving_side_evidence(is_left: bool, *, analyst_confirmed: bool = True) -> Evidence:
    """Build a driving-side constraint from an analyst determination."""
    if is_left:
        scores = dict.fromkeys(LEFT_DRIVING, 1.0)
        floor = 0.02
    else:
        # Right-driving is the complement; express it as a low floor on the
        # left-driving set rather than enumerating 160 countries.
        scores = dict.fromkeys(LEFT_DRIVING, 0.02)
        floor = 1.0
    conf = Confidence.HIGH if analyst_confirmed else Confidence.LOW
    return Evidence(
        id="analyst.driving_side", analyzer="analyst",
        title=f"Driving side (analyst): {'LEFT' if is_left else 'RIGHT'}",
        detail=("Analyst-determined driving side. "
                + (f"{len(LEFT_DRIVING)} countries drive on the left."
                   if is_left else
                   "Excludes the left-driving countries.")),
        confidence=conf, tags=["driving", "analyst"],
        constraints=[GeoConstraint(
            kind=ConstraintKind.COUNTRIES, country_scores=scores,
            floor=floor, confidence=conf, note="driving side",
        )],
    )
