"""Measuring how often a reference corpus names a place it does not contain.

Calibration is leave-one-SITE-out rather than leave-one-image-out, and the
distinction is the whole point. Holding out one image leaves its site in the
index, so it only measures ranking among places the corpus already knows. In
practice the photographed place is usually absent from every corpus on disk,
and what matters is whether the tool says so.

Two figures result:

* fabrication rate -- queries whose site was removed wholesale. Every
  acceptance there identifies a place the corpus does not contain, and this
  is the number a corpus should be judged by.
* recall -- queries whose site remains. It measures how much genuine signal
  the thresholds throw away, which is the price of that caution.

Results are saved beside the index so the workbench can show them, and are
marked stale once the index changes underneath them.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import corpus, match
from . import index as index_mod

# Above this, a corpus is flagged as untrustworthy. The thresholds in match.py
# held fabrication to 2.9% on a well-covered corpus, so 5% leaves headroom for
# harder areas without accepting a corpus that guesses routinely.
ACCEPTABLE_FABRICATION = 0.05


class _Subset:
    """An index restricted to some of its rows, for holding sites out."""

    def __init__(self, descriptors: np.ndarray, records: list[dict[str, Any]]):
        self.descriptors = descriptors
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def search(self, query: np.ndarray, top_k: int = 25) -> list[tuple[float, dict]]:
        if not self.records:
            return []
        sims = self.descriptors @ query
        k = min(top_k, sims.size)
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(float(sims[i]), self.records[i]) for i in idx]


def _subset(idx, mask: np.ndarray) -> _Subset:
    return _Subset(idx.descriptors[mask],
                   [r for r, keep in zip(idx.records, mask, strict=True) if keep])


def evaluate(idx, per_site: int = 6) -> dict[str, Any]:
    """Fabrication rate and recall for an index, without touching disk."""
    site_ids = sorted({r["site_id"] for r in idx.records})
    open_n = fabricated = closed_n = correct = 0

    for site in site_ids:
        others = np.array([r["site_id"] != site for r in idx.records])
        held = [i for i, r in enumerate(idx.records) if r["site_id"] == site]
        if len(held) < 2:
            continue
        absent = _subset(idx, others)
        for qi in held[:per_site]:
            query = idx.descriptors[qi]
            open_n += 1
            fabricated += bool(match.match(absent, query).accepted)

            present = others.copy()
            present[held] = True
            present[qi] = False
            found = match.match(_subset(idx, present), query)
            closed_n += 1
            correct += bool(found.accepted and found.matches
                            and found.matches[0].site_id == site)

    base: dict[str, Any] = {
        "images": len(idx.records),
        "sites": len(site_ids),
        "open_queries": open_n,
        "fabricated": fabricated,
        "closed_queries": closed_n,
        "correct": correct,
        "thresholds": {"min_similarity": match.MIN_SIMILARITY,
                       "min_margin": match.MIN_MARGIN,
                       "min_ratio": match.MIN_RATIO},
    }
    if not open_n:
        return {**base, "fabrication_rate": None, "recall": None,
                "verdict": "insufficient",
                "message": ("Every site has fewer than two images, so none can "
                            "be held out. Harvest more images per site, then "
                            "calibrate again.")}

    rate = fabricated / open_n
    recall = correct / closed_n
    if rate > ACCEPTABLE_FABRICATION:
        return {**base, "fabrication_rate": rate, "recall": recall,
                "verdict": "too_high",
                "message": (f"Named a place the corpus does not contain in "
                            f"{rate:.0%} of {open_n} tests. Harvest denser "
                            "coverage per site before trusting its matches.")}
    return {**base, "fabrication_rate": rate, "recall": recall, "verdict": "ok",
            "message": (f"Named an absent place in {rate:.1%} of {open_n} tests, "
                        f"and found the right site {recall:.0%} of the time when "
                        "it was present. The gap is deliberate: uncertain "
                        "matches are refused rather than guessed.")}


def _path(area: str) -> Path:
    return corpus.area_dir(area) / "calibration.json"


def run(area: str, per_site: int = 6) -> dict[str, Any]:
    """Calibrate a corpus on disk and save the result beside its index."""
    idx = index_mod.load(area)
    if idx is None:
        raise FileNotFoundError(f"No index for corpus {area!r}. Build it first.")
    result = {**evaluate(idx, per_site), "area": area,
              "calibrated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _path(area).write_text(json.dumps(result, indent=1))
    return result


def load(area: str) -> dict[str, Any] | None:
    try:
        return json.loads(_path(area).read_text())
    except (OSError, json.JSONDecodeError):
        return None
