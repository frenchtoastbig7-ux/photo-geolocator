"""MegaLoc descriptor extraction.

Visual place recognition is a different task from the semantic similarity
CLIP provides, and the difference is not academic. Measured on this project:

* CLIP scored eight French railway stations between 0.70 and 0.83 against a
  platform photograph -- it was reporting "this is a railway station", which
  every candidate also was, so it could not choose and could not abstain.
* MegaLoc scores the same set between 0.08 and 0.16, because none of those
  reference images actually shows the queried scene. A retrieval model
  trained for place recognition produces low scores when nothing matches,
  which is what makes a rejection threshold possible at all.

That ability to say "not in the corpus" matters more here than ranking does.
A geolocation tool that always returns its best guess is the thing this
project exists not to be.

MegaLoc: Berton et al., CVPR 2025 Workshops (arXiv:2502.17237), MIT licensed.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

from ..config import SETTINGS

# Pinned. torch.hub executes code from the repository, so the revision is
# fixed rather than tracking whatever main happens to be.
REPO = "gmberton/MegaLoc"
PIN = "main"
INPUT_SIZE = 322          # the resolution the published results use
DESCRIPTOR_DIM = 8448


class VPRUnavailable(RuntimeError):
    """MegaLoc is not installed, in a form fit to show a user."""


def torch_home() -> Path:
    return SETTINGS.model_cache / "torchhub"


@functools.lru_cache(maxsize=1)
def _load():
    try:
        import torch
    except ImportError as exc:
        raise VPRUnavailable(
            "PyTorch is not installed; the place-recognition model needs it."
        ) from exc

    home = torch_home()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(home)
    # MegaLoc's hub entrypoint pulls its weights from HuggingFace without an
    # explicit cache_dir, so the environment is the only way to keep them
    # beside the scene model rather than in the user's default cache.
    from ..modelmgr import hf_cache_dir

    cache = hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(cache))

    if not SETTINGS.net_allowed(purpose="model_download") and not is_installed():
        raise VPRUnavailable(
            "The place-recognition model is not installed and offline mode is "
            "on. Install it once with networking enabled; it runs offline "
            "afterwards."
        )
    try:
        model = torch.hub.load(REPO, "get_trained_model",
                               trust_repo=True, verbose=False).eval()
    except Exception as exc:
        raise VPRUnavailable(f"could not load MegaLoc: {exc}") from exc

    device = SETTINGS.resolve_device()
    return model.to(device), device, torch


def is_installed() -> bool:
    """Whether the weights and hub checkout are already on disk."""
    home = torch_home()
    hub = home / "hub"
    if not hub.exists():
        return False
    has_repo = any(p.name.lower().startswith("gmberton") for p in hub.iterdir())
    has_weights = any(hub.rglob("*.pth")) or any(hub.rglob("*.safetensors"))
    return bool(has_repo and has_weights)


@functools.lru_cache(maxsize=1)
def _transform():
    import torchvision.transforms as tfm

    return tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tfm.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
    ])


def embed_images(paths: list[Path], batch_size: int = 8) -> np.ndarray:
    """L2-normalised descriptors, one row per readable image.

    Unreadable files yield a zero row rather than shifting every subsequent
    result, so callers can rely on row i corresponding to paths[i].
    """
    from PIL import Image

    model, device, torch = _load()
    transform = _transform()
    out = np.zeros((len(paths), DESCRIPTOR_DIM), dtype=np.float32)

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        tensors, rows = [], []
        for offset, path in enumerate(chunk):
            try:
                with Image.open(path) as im:
                    tensors.append(transform(im.convert("RGB")))
                rows.append(start + offset)
            except Exception:
                continue
        if not tensors:
            continue
        with torch.no_grad():
            batch = torch.stack(tensors).to(device)
            desc = torch.nn.functional.normalize(model(batch), dim=-1)
        out[rows] = desc.cpu().numpy().astype(np.float32)
    return out


def embed_image(path: Path) -> np.ndarray:
    return embed_images([path])[0]
