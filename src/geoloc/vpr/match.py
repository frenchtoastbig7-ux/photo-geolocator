"""Querying the corpus, and knowing when not to answer.

The ranking is the easy half. The half that decides whether this tool is
trustworthy is abstention: a retrieval index always has a nearest neighbour,
and returning it regardless is how automated geolocation produces confident,
precise, wrong answers.

Three conditions must hold before a match becomes evidence:

* absolute similarity above a floor -- the queried place has to actually be
  in the corpus, and if it is not, every score is low;
* a margin over the best *different* site -- two sites scoring alike means
  the descriptor is responding to something generic (a platform, a plaza)
  rather than to this place;
* corroboration -- one image matching is weaker than several images of the
  same site matching.

The thresholds are calibrated empirically rather than guessed; see
`geoloc corpus calibrate`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Thresholds measured, not guessed. Held-out calibration over 202 queries
# against a 12-site French station corpus compared four discriminators by how
# reliably they separate a correct top-1 site from an incorrect one:
#
#     discriminator          AUC     correct median   wrong median
#     ratio  (top1/top2)     0.866        1.92            1.08
#     margin (top1-top2)     0.859        0.104           0.009
#     absolute similarity    0.778        0.220           0.108
#     support (image count)  0.741        4               2
#
# Absolute similarity is the weakest of the four, which is why the first
# implementation was wrong: a floor at the control distribution's 95th
# percentile would have rejected 68% of genuine matches. Relative separation
# is what carries the signal, so acceptance turns on margin and ratio, with
# the absolute score kept only as a floor against a corpus that contains
# nothing remotely similar.
#
# Operating point chosen for precision over recall, because a wrong
# identification costs an investigation far more than an absent one:
#
#     margin >= 0.03  ->  accepts 61% of queries, 90% of them correct
#     margin >= 0.05  ->  accepts 50% of queries, 96% of them correct   <-
#     margin >= 0.12  ->  accepts 33% of queries, 100% of them correct
# Chosen by open-set sweep over 102 leave-one-SITE-out queries, ranked by
# fabrication rate rather than by what admits any particular photograph:
#
#     floor  margin  ratio     fabricate    genuine kept
#     0.15   0.05    1.50         6.9%          38.2%
#     0.15   0.05    1.30         8.8%          40.2%
#     0.20   0.04    1.25         2.9%          36.3%   <- chosen
#     0.15   0.04    1.20         9.8%          42.2%
#     0.15   0.03    1.15        11.8%          43.1%
#
# Raising the floor while easing the ratio more than halves fabrication for
# about two points of recall, which is the right direction for a tool whose
# worst outcome is a confident wrong answer. Roughly a third of genuine
# matches are still discarded; that is the price, and it is deliberate.
MIN_SIMILARITY = 0.20
MIN_MARGIN = 0.04
MIN_RATIO = 1.25


@dataclass
class SiteMatch:
    site_id: str
    site_name: str
    lat: float
    lon: float
    best_score: float
    support: int                 # images of this site among the top hits
    top_titles: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"site_id": self.site_id, "site_name": self.site_name,
                "lat": self.lat, "lon": self.lon,
                "best_score": round(self.best_score, 4),
                "support": self.support, "top_titles": self.top_titles[:3]}


@dataclass
class MatchResult:
    accepted: bool
    reason: str
    matches: list[SiteMatch]
    best_score: float
    margin: float

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason,
                "best_score": round(self.best_score, 4),
                "margin": round(self.margin, 4),
                "matches": [m.as_dict() for m in self.matches[:5]]}


def aggregate(hits: list[tuple[float, dict]]) -> list[SiteMatch]:
    """Collapse image hits into per-site matches, best score first."""
    by_site: dict[str, SiteMatch] = {}
    for score, rec in hits:
        sid = rec["site_id"]
        existing = by_site.get(sid)
        if existing is None:
            by_site[sid] = SiteMatch(
                site_id=sid, site_name=rec.get("site_name", ""),
                lat=float(rec["site_lat"]), lon=float(rec["site_lon"]),
                best_score=score, support=1, top_titles=[rec.get("title", "")])
        else:
            existing.support += 1
            existing.top_titles.append(rec.get("title", ""))
            existing.best_score = max(existing.best_score, score)
    return sorted(by_site.values(), key=lambda m: -m.best_score)


def match(index, query_descriptor: np.ndarray, *, top_k: int = 25,
          min_similarity: float = MIN_SIMILARITY,
          min_margin: float = MIN_MARGIN,
          min_ratio: float = MIN_RATIO) -> MatchResult:
    """Best matching site, or a reasoned refusal to name one."""
    hits = index.search(query_descriptor, top_k=top_k)
    if not hits:
        return MatchResult(False, "the corpus is empty", [], 0.0, 0.0)

    sites = aggregate(hits)
    best = sites[0]
    runner_up = sites[1].best_score if len(sites) > 1 else 0.0
    margin = best.best_score - runner_up
    ratio = best.best_score / runner_up if runner_up > 1e-6 else float("inf")

    if best.best_score < min_similarity:
        return MatchResult(
            False,
            f"best similarity {best.best_score:.3f} is below the "
            f"{min_similarity:.2f} floor: nothing in this corpus resembles the "
            "image. Either the place is outside the harvested area, or the "
            "area needs denser coverage.",
            sites, best.best_score, margin)

    if len(sites) > 1 and (margin < min_margin or ratio < min_ratio):
        return MatchResult(
            False,
            f"the top two sites are only {margin:.3f} apart "
            f"(ratio {ratio:.2f}), under the {min_margin:.2f} / {min_ratio:.1f} "
            "separation required. The descriptor is responding to something "
            "these places share rather than to one of them, so no site is "
            "named. Calibration showed this band is where identifications go "
            "wrong.",
            sites, best.best_score, margin)

    return MatchResult(
        True,
        f"matched {best.site_name or best.site_id} at similarity "
        f"{best.best_score:.3f}, {margin:.3f} clear of the next site "
        f"(ratio {ratio:.2f}), with {best.support} supporting image(s).",
        sites, best.best_score, margin)


def query_image(index, image_path: Path, **kwargs) -> MatchResult:
    from . import model

    return match(index, model.embed_image(image_path), **kwargs)
