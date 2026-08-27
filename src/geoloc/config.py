"""Runtime configuration and the global network kill-switch.

The single most important invariant in this application: the analysed image
never leaves the host. Outbound traffic is limited to *derived* textual
queries (place-feature searches against Overpass/Nominatim) and, on first
run, model-weight downloads. `OFFLINE` disables all of it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass
class Settings:
    # --- network posture -------------------------------------------------
    offline: bool = field(default_factory=lambda: _env_bool("GEOLOC_OFFLINE", False))
    """Hard kill-switch. When True no socket is opened for any reason."""

    allow_model_download: bool = field(
        default_factory=lambda: _env_bool("GEOLOC_ALLOW_MODEL_DOWNLOAD", True)
    )
    """Permit one-time HuggingFace weight fetches. Independent of `offline`
    only in the sense that `offline` overrides it."""

    overpass_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "GEOLOC_OVERPASS", "https://overpass-api.de/api/interpreter"
        )
    )
    nominatim_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "GEOLOC_NOMINATIM", "https://nominatim.openstreetmap.org/search"
        )
    )
    user_agent: str = "geoloc-osint/0.1 (local research tool)"
    http_timeout: float = 45.0

    # --- storage ---------------------------------------------------------
    case_dir: Path = field(default_factory=lambda: _env_path("GEOLOC_CASE_DIR", PROJECT_ROOT / "cases"))
    model_cache: Path = field(
        default_factory=lambda: _env_path("GEOLOC_MODEL_CACHE", Path.home() / ".cache" / "geoloc")
    )

    # --- analysis knobs --------------------------------------------------
    grid_step: float = float(os.environ.get("GEOLOC_GRID_STEP", "0.5"))
    """Degrees per fusion-grid cell. 0.5 deg -> 360x720 = 259k cells."""

    clip_model: str = os.environ.get("GEOLOC_CLIP_MODEL", "geolocal/StreetCLIP")
    device: str = os.environ.get("GEOLOC_DEVICE", "auto")

    # --- guardrails ------------------------------------------------------
    blur_faces_in_exports: bool = field(
        default_factory=lambda: _env_bool("GEOLOC_BLUR_FACES", True)
    )

    def net_allowed(self, *, purpose: str = "query") -> bool:
        """Return True if an outbound request for `purpose` is permitted."""
        if self.offline:
            return False
        if purpose == "model_download":
            return self.allow_model_download
        return True

    def ensure_dirs(self) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


SETTINGS = Settings()


class OfflineError(RuntimeError):
    """Raised when code attempts network access while offline mode is on."""
