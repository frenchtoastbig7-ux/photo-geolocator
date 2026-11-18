"""On-demand scene-model installation.

PyTorch is bundled into the .app, but StreetCLIP's weights (1.7 GB) are not:
they would push the disk image past what is practical to distribute. The app
is fully useful without them -- metadata, OCR, solar geometry and fusion all
work -- and this module lets the analyst install the scene model from inside
the app when they want it.

The revision is pinned. The repository has both a `.bin` revision and a
newer safetensors one; letting transformers resolve it dynamically fetched
*both*, turning a 1.7 GB download into 3.2 GB on disk.
"""
from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import SETTINGS

MODEL_ID = "geolocal/StreetCLIP"
MODEL_REVISION = "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"

# Weights, config and tokeniser only. The repo also carries two demo
# photographs and a README that serve no runtime purpose.
ALLOW_PATTERNS = ["*.json", "*.txt", "pytorch_model.bin"]

# Used only to render a progress bar before the real total is known.
APPROX_TOTAL_BYTES = 1_716_000_000


@dataclass
class DownloadState:
    state: str = "idle"          # idle | running | done | error | cancelled
    downloaded: int = 0
    total: int = APPROX_TOTAL_BYTES
    error: str = ""
    _thread: threading.Thread | None = field(default=None, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict[str, Any]:
        pct = (self.downloaded / self.total * 100.0) if self.total else 0.0
        return {
            "state": self.state,
            "downloaded": self.downloaded,
            "total": self.total,
            "percent": round(min(pct, 100.0), 1),
            "error": self.error,
        }


_STATE = DownloadState()
_LOCK = threading.Lock()


def hf_cache_dir() -> Path:
    """Where HuggingFace artefacts live for this install."""
    return SETTINGS.model_cache / "huggingface"


def _repo_cache_dir() -> Path:
    slug = "models--" + MODEL_ID.replace("/", "--")
    return hf_cache_dir() / slug


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def snapshot_path() -> Path | None:
    """Local snapshot directory if the model is fully cached, else None."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        p = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=str(hf_cache_dir()),
            allow_patterns=ALLOW_PATTERNS, local_files_only=True,
        )
    except Exception:
        return None
    path = Path(p)
    # snapshot_download succeeds against a partial cache, so confirm the
    # weights are actually present rather than trusting the call.
    weights = path / "pytorch_model.bin"
    if not weights.exists():
        return None
    try:
        if weights.stat().st_size < 1_000_000_000:
            return None
    except OSError:
        return None
    return path


def is_installed() -> bool:
    return snapshot_path() is not None


def status() -> dict[str, Any]:
    installed = is_installed()
    with _LOCK:
        prog = _STATE.as_dict()
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION[:12],
        "installed": installed,
        "torch_available": torch_available(),
        "cache_dir": str(hf_cache_dir()),
        "approx_bytes": APPROX_TOTAL_BYTES,
        "disk_free_bytes": shutil.disk_usage(str(SETTINGS.model_cache.parent)).free
        if SETTINGS.model_cache.parent.exists() else None,
        "progress": prog,
    }


def _download_worker() -> None:
    from huggingface_hub import snapshot_download

    repo_dir = _repo_cache_dir()
    baseline = _dir_size(repo_dir)

    stop_polling = threading.Event()

    def poll() -> None:
        while not stop_polling.wait(0.5):
            with _LOCK:
                _STATE.downloaded = max(0, _dir_size(repo_dir) - baseline)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()

    try:
        # Resolve the true byte total so the bar is accurate rather than
        # anchored to a constant.
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION,
                                      files_metadata=True)
            import fnmatch
            total = sum(
                f.size or 0 for f in info.siblings
                if any(fnmatch.fnmatch(f.rfilename, pat) for pat in ALLOW_PATTERNS)
            )
            if total:
                with _LOCK:
                    _STATE.total = total
        except Exception:
            pass

        snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=str(hf_cache_dir()),
            allow_patterns=ALLOW_PATTERNS, max_workers=4,
        )
    except Exception as exc:
        with _LOCK:
            _STATE.state = "error"
            _STATE.error = f"{type(exc).__name__}: {exc}"
        return
    finally:
        stop_polling.set()

    with _LOCK:
        if is_installed():
            _STATE.state = "done"
            _STATE.downloaded = _STATE.total
        else:
            _STATE.state = "error"
            _STATE.error = "download finished but the weights are still incomplete"


def start_download() -> dict[str, Any]:
    """Begin fetching the weights on a background thread."""
    if not SETTINGS.net_allowed(purpose="model_download"):
        return {"ok": False,
                "reason": "Offline mode is on. Turn it off to install the "
                          "scene model, then turn it back on."}
    if is_installed():
        return {"ok": True, "reason": "already installed"}

    with _LOCK:
        if _STATE.state == "running":
            return {"ok": True, "reason": "already downloading"}
        _STATE.state = "running"
        _STATE.downloaded = 0
        _STATE.error = ""
        _STATE._cancel.clear()

    SETTINGS.model_cache.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=_download_worker, daemon=True, name="model-download")
    with _LOCK:
        _STATE._thread = t
    t.start()
    return {"ok": True, "reason": "started"}


def remove() -> dict[str, Any]:
    """Delete the cached weights to reclaim disk space."""
    repo_dir = _repo_cache_dir()
    if not repo_dir.exists():
        return {"ok": True, "freed_bytes": 0}
    freed = _dir_size(repo_dir)
    shutil.rmtree(repo_dir, ignore_errors=True)
    with _LOCK:
        _STATE.state = "idle"
        _STATE.downloaded = 0
    return {"ok": True, "freed_bytes": freed}
