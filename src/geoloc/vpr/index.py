"""On-disk descriptor index for a harvested area."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import corpus, model


@dataclass
class VPRIndex:
    area: str
    descriptors: np.ndarray            # (N, D) L2-normalised, float32
    records: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def sites(self) -> set[str]:
        return {r["site_id"] for r in self.records}

    def search(self, query: np.ndarray, top_k: int = 20) -> list[tuple[float, dict]]:
        if not len(self.records):
            return []
        sims = self.descriptors @ query
        k = min(top_k, sims.size)
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(float(sims[i]), self.records[i]) for i in idx]


def index_path(area: str) -> Path:
    return corpus.area_dir(area) / "index.npz"


def build(area: str, *, progress=None, on_progress=None,
          batch_size: int = 8) -> VPRIndex:
    """Embed every harvested image for an area and persist the descriptors.

    Incremental: images already embedded are carried over, so re-running
    after extending a corpus only costs the new files.
    """
    records = corpus.load_manifest(area)
    if not records:
        raise FileNotFoundError(
            f"no harvested imagery for area {area!r} -- run a corpus build first")

    existing: dict[str, np.ndarray] = {}
    path = index_path(area)
    if path.exists():
        try:
            blob = np.load(path, allow_pickle=False)
            prior = json.loads(blob["records"].item()) if "records" in blob else []
            for rec, vec in zip(prior, blob["descriptors"], strict=False):
                existing[rec["path"]] = vec
        except Exception:
            existing = {}

    todo = [r for r in records if r["path"] not in existing]
    if todo and progress:
        progress(f"embedding {len(todo)} new image(s)")
    if todo:
        paths = [Path(r["path"]) for r in todo]
        vectors = model.embed_images(paths, batch_size=batch_size,
                                    progress=on_progress)
        for rec, vec in zip(todo, vectors, strict=True):
            existing[rec["path"]] = vec

    keep = [r for r in records if r["path"] in existing]
    # A zero row means the file could not be read; carrying it would let an
    # unreadable image match everything at similarity zero.
    matrix = np.stack([existing[r["path"]] for r in keep]).astype(np.float32)
    good = np.linalg.norm(matrix, axis=1) > 0.5
    matrix, keep = matrix[good], [r for r, ok in zip(keep, good, strict=True) if ok]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, descriptors=matrix,
                        records=np.array(json.dumps(keep)))
    return VPRIndex(area=area, descriptors=matrix, records=keep)


def load(area: str) -> VPRIndex | None:
    path = index_path(area)
    if not path.exists():
        return None
    try:
        blob = np.load(path, allow_pickle=False)
        records = json.loads(blob["records"].item())
        return VPRIndex(area=area, descriptors=blob["descriptors"], records=records)
    except Exception:
        return None


def indexed_count(area: str) -> int:
    """Number of indexed images, without loading the descriptors.

    The workbench lists corpora on every refresh; reading only the records
    entry of the archive keeps that from decompressing thousands of 8448-wide
    descriptors each time.
    """
    path = index_path(area)
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=False) as blob:
            return len(json.loads(blob["records"].item()))
    except Exception:
        return 0
