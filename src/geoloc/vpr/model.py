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
    """Load MegaLoc, touching the network only when it is not yet installed.

    Once the code checkout and the weights are on disk the model is built from
    them directly, bypassing torch.hub and the upstream hubconf. Both reach
    the network on every load even when everything is cached: torch.hub
    validates the repository against the GitHub API unless told not to, and
    the hubconf calls hf_hub_download, which asks HuggingFace whether a newer
    file exists and pings it for download statistics. Offline mode must mean
    no outbound traffic, not merely no downloads.
    """
    try:
        import torch
    except ImportError as exc:
        raise VPRUnavailable(
            "PyTorch is not installed; the place-recognition model needs it."
        ) from exc

    if is_installed():
        model = _load_local()
    elif not SETTINGS.net_allowed(purpose="model_download"):
        raise VPRUnavailable(
            "The place-recognition model is not installed and offline mode is "
            "on. Install it once with networking enabled; it runs offline "
            "afterwards."
        )
    else:
        home = torch_home()
        home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(home)
        cache = _hf_cache()
        cache.mkdir(parents=True, exist_ok=True)
        # The hubconf downloads weights without a cache_dir, so the
        # environment is the only way to keep them beside the scene model.
        os.environ.setdefault("HF_HUB_CACHE", str(cache))
        try:
            model = torch.hub.load(REPO, "get_trained_model",
                                   trust_repo=True, verbose=False).eval()
        except Exception as exc:
            raise VPRUnavailable(f"could not install MegaLoc: {exc}") from exc

    device = SETTINGS.resolve_device()
    return model.to(device), device, torch


def _hf_cache() -> Path:
    from ..modelmgr import hf_cache_dir

    return hf_cache_dir()


CHECKOUT_DIR = "gmberton_MegaLoc_main"
WEIGHTS_REPO_DIR = "models--gberton--MegaLoc"
# The genuine file is ~870 MB. Anything far smaller is an interrupted download,
# which would fail strict state-dict loading with an unhelpful shape error.
MIN_WEIGHT_BYTES = 500_000_000


def local_checkout() -> Path | None:
    """The MegaLoc code fetched by torch.hub, if present."""
    repo = torch_home() / "hub" / CHECKOUT_DIR
    return repo if (repo / "megaloc_model.py").is_file() else None


def weights_path() -> Path | None:
    """The cached MegaLoc weights, newest snapshot first.

    Snapshot entries are symlinks into the blob store, so `is_file` follows
    them and rejects a snapshot whose blob never finished downloading.
    """
    snapshots = _hf_cache() / WEIGHTS_REPO_DIR / "snapshots"
    found = []
    for f in snapshots.glob("*/model.safetensors"):
        try:
            if f.is_file() and f.stat().st_size >= MIN_WEIGHT_BYTES:
                found.append(f)
        except OSError:
            continue
    return max(found, key=lambda f: f.stat().st_mtime) if found else None


def is_installed() -> bool:
    """Whether the code and weights needed to load offline are both present.

    The earlier check looked for weights inside the torch.hub directory, where
    MegaLoc never stores them: its hubconf puts them in the HuggingFace cache.
    It therefore always reported "not installed", which silently disabled
    visual matching whenever offline mode was on.
    """
    return local_checkout() is not None and weights_path() is not None


def _load_local():
    """Build MegaLoc from files on disk, with no network access at all."""
    import importlib.util

    from safetensors.torch import load_file

    checkout, weights = local_checkout(), weights_path()
    if checkout is None or weights is None:
        raise VPRUnavailable("MegaLoc files on disk are incomplete.")

    spec = importlib.util.spec_from_file_location(
        "_geoloc_megaloc_model", checkout / "megaloc_model.py")
    if spec is None or spec.loader is None:
        raise VPRUnavailable(f"cannot import MegaLoc from {checkout}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.MegaLoc()
    try:
        model.load_state_dict(load_file(str(weights)))
    except Exception as exc:
        raise VPRUnavailable(f"MegaLoc weights at {weights} did not load: {exc}") from exc
    return model.eval()


@functools.lru_cache(maxsize=1)
def _transform():
    import torchvision.transforms as tfm

    return tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tfm.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
    ])


def embed_images(paths: list[Path], batch_size: int = 8,
                 progress=None) -> np.ndarray:
    """L2-normalised descriptors, one row per readable image.

    Unreadable files yield a zero row rather than shifting every subsequent
    result, so callers can rely on row i corresponding to paths[i].
    """
    from PIL import Image

    model, device, torch = _load()
    transform = _transform()
    out = np.zeros((len(paths), DESCRIPTOR_DIM), dtype=np.float32)

    for start in range(0, len(paths), batch_size):
        if progress:
            progress(start, len(paths))
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
    if progress:
        progress(len(paths), len(paths))
    return out


def embed_image(path: Path) -> np.ndarray:
    return embed_images([path])[0]
