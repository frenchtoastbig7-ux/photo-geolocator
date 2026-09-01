"""Does the file's own metadata agree with what the picture shows?

A GPS tag is trivially editable, so treating one as ground truth is exactly
how a doctored image passes review. The check implemented here is
adversarial: the scene-derived posterior is recomputed with every
metadata-derived constraint removed, and the tagged coordinates are then
scored against that independent belief.

The distinction matters. Fusing the tag and then observing that the answer
sits on the tag proves nothing -- the tag put it there. Excluding it first
means the question becomes "would the photograph's own content have led us
here?", which is the question an operator actually needs answered.

Absence of conflict is not proof of authenticity: a photograph with no
legible text and no recognisable structure cannot corroborate anything, and
this reports that as unverifiable rather than as agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geo.grid import WorldGrid, haversine_km
from .models import Evidence

# Distance beyond which a tag and the scene evidence are telling different
# stories. Generous: content evidence is usually town-level at best, so a
# tighter bound would flag honest imprecision as fraud.
CONFLICT_KM = 150.0
STRONG_CONFLICT_KM = 750.0

# Below this share of the peak, the content is not pointing at the tag.
WEAK_SUPPORT = 0.05


@dataclass
class MetadataVerdict:
    verdict: str = "unverifiable"      # consistent | conflict | unverifiable | no_metadata
    has_gps: bool = False
    gps: tuple[float, float] | None = None
    content_best: tuple[float, float] | None = None
    distance_km: float | None = None
    gps_support: float | None = None
    """Probability the scene evidence assigns the tagged cell, relative to
    its own strongest cell. 1.0 means the content points exactly there; near
    zero means the content points somewhere else entirely.

    Deliberately NOT a percentile. A posterior concentrated on one country is
    near-zero across almost every cell on Earth, so a percentile flatters any
    point at all: a German street scene tagged with Tokyo coordinates scored
    in the 93rd percentile and was reported as consistent, 8,900 km from
    where the image actually pointed."""
    headline: str = ""
    detail: str = ""
    indicators: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "has_gps": self.has_gps,
            "gps": list(self.gps) if self.gps else None,
            "content_best": list(self.content_best) if self.content_best else None,
            "distance_km": (round(self.distance_km, 1)
                            if self.distance_km is not None else None),
            "gps_support": (round(self.gps_support, 5)
                            if self.gps_support is not None else None),
            "headline": self.headline, "detail": self.detail,
            "indicators": self.indicators,
        }


def _is_metadata_evidence(ev: Evidence) -> bool:
    return ev.analyzer == "metadata"


def verify(evidence: list[Evidence], grid: WorldGrid, heatmaps: dict,
           *, use_habitation_prior: bool = True) -> MetadataVerdict:
    """Score the file's GPS tag against the scene evidence alone."""
    from .geo import fusion

    gps_ev = next((e for e in evidence
                   if e.id == "meta.gps" and e.raw.get("lat") is not None), None)

    # Tamper indicators worth surfacing whether or not a tag is present.
    indicators: list[str] = []
    for ev in evidence:
        if ev.analyzer != "metadata":
            continue
        if ev.id == "meta.stripped":
            indicators.append("Metadata has been stripped or re-encoded.")
        if ev.id == "meta.software" and ev.raw.get("software"):
            indicators.append(f"Processed by {ev.raw['software']}.")
        if ev.id == "meta.platform_filename":
            indicators.append("Filename matches a social-media export pattern.")

    if gps_ev is None:
        return MetadataVerdict(
            verdict="no_metadata", has_gps=False,
            headline="No GPS tag to verify.",
            detail=("The file carries no coordinates, so there is nothing to "
                    "corroborate or contradict. Location rests entirely on "
                    "scene content."),
            indicators=indicators)

    lat, lon = float(gps_ev.raw["lat"]), float(gps_ev.raw["lon"])

    # The adversarial step: rebuild belief from the scene alone.
    scene_only = [e for e in evidence if not _is_metadata_evidence(e)]
    informative = [e for e in scene_only if e.constraints]
    if not informative:
        return MetadataVerdict(
            verdict="unverifiable", has_gps=True, gps=(lat, lon),
            headline="GPS tag present, but nothing in the image can corroborate it.",
            detail=("No scene evidence carries a location constraint, so the "
                    "tag can be neither supported nor contradicted. Treat it "
                    "as an unverified claim, not as a fact."),
            indicators=indicators)

    post, g = fusion.fuse(informative, grid=grid, heatmaps=heatmaps,
                          use_habitation_prior=use_habitation_prior)
    row, col = g.cell_of(lat, lon)
    tagged_mass = float(post[row, col])
    peak_mass = float(post.max())
    support = tagged_mass / peak_mass if peak_mass > 0 else 0.0

    peak = np.unravel_index(int(np.argmax(post)), post.shape)
    best = (float(g.lat2d[peak]), float(g.lon2d[peak]))
    distance = haversine_km(lat, lon, best[0], best[1])

    common = {"has_gps": True, "gps": (lat, lon), "content_best": best,
              "distance_km": distance, "gps_support": support,
              "indicators": indicators}

    # Support is the discriminator; distance alone would flag honest
    # imprecision on a coarse posterior, and support alone would accept a tag
    # sitting on a weak secondary mode.
    if support < WEAK_SUPPORT and distance >= CONFLICT_KM:
        severity = ("cannot be reconciled with"
                    if distance >= STRONG_CONFLICT_KM else "disagrees with")
        return MetadataVerdict(
            verdict="conflict",
            headline=(f"GPS tag {severity} the image content — "
                      f"{distance:,.0f} km apart."),
            detail=(f"The tag claims {lat:.5f}, {lon:.5f}. Excluding the tag "
                    f"entirely, the image's own content points to "
                    f"{best[0]:.3f}, {best[1]:.3f}, and assigns the tagged "
                    f"location {support:.2%} of the confidence it gives its "
                    "own best answer. "
                    + ("A discrepancy of this size is not imprecision: treat "
                       "the coordinates as fabricated or copied from another "
                       "image until independently corroborated."
                       if distance >= STRONG_CONFLICT_KM else
                       "Worth resolving before relying on either.")),
            **common)

    return MetadataVerdict(
        verdict="consistent",
        headline=f"GPS tag is consistent with the image content ({distance:,.0f} km apart).",
        detail=(f"Excluding the tag, the image's own content assigns the "
                f"tagged location {support:.0%} of the confidence it gives "
                "its best answer, so the photograph independently supports "
                "the coordinates. "
                "This is corroboration, not proof: content evidence is coarse, "
                "and a plausible tag can still be fabricated."),
        **common)
