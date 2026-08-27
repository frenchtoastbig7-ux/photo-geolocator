"""Core data model.

The whole application is a pipeline that turns one image into a list of
`Evidence`, each of which may carry one or more `GeoConstraint`s. Fusion
combines the constraints into a posterior over the globe; `Candidate`s are
extracted from that posterior. Everything is serialisable so a case can be
saved, reopened, and audited.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Confidence(StrEnum):
    """Qualitative confidence, deliberately coarse.

    OSINT reporting conventions favour stated confidence bands over false
    precision. These map to numeric weights in fusion.
    """

    CERTAIN = "certain"      # e.g. an unmodified embedded GPS tag
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"

    @property
    def weight(self) -> float:
        return {
            "certain": 8.0,
            "high": 3.0,
            "medium": 1.5,
            "low": 0.6,
            "speculative": 0.2,
        }[self.value]


class ConstraintKind(StrEnum):
    POINT = "point"              # a specific coordinate, with a radius
    COUNTRIES = "countries"      # allow/deny list of ISO-3166 alpha-2 codes
    LAT_BAND = "lat_band"        # latitude range (solar geometry, biome)
    LON_BAND = "lon_band"        # longitude range (timezone offsets)
    BBOX = "bbox"                # rectangular region
    HEATMAP = "heatmap"          # dense per-cell log-likelihood (ML estimator)


class GeoConstraint(BaseModel):
    """A single geographic restriction derived from one piece of evidence."""

    kind: ConstraintKind
    confidence: Confidence = Confidence.MEDIUM
    # POINT
    lat: float | None = None
    lon: float | None = None
    radius_km: float | None = None
    # COUNTRIES  (iso2 -> relative likelihood, unlisted countries get `floor`)
    country_scores: dict[str, float] | None = None
    floor: float = 0.02
    """Likelihood assigned to regions not named by the constraint. Keeping
    this above zero prevents any single soft cue from vetoing the truth."""
    # BANDS / BBOX
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    softness_deg: float = 3.0
    """Gaussian falloff applied outside a band/bbox edge, in degrees."""
    # HEATMAP -- flattened row-major log-likelihood matching the fusion grid
    heatmap_ref: str | None = None

    note: str = ""


class Evidence(BaseModel):
    """One observation made by one analyzer."""

    id: str
    analyzer: str
    title: str
    detail: str = ""
    confidence: Confidence = Confidence.MEDIUM
    constraints: list[GeoConstraint] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    """Analyzer-specific payload retained verbatim for auditability."""
    tags: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    """A ranked location hypothesis extracted from the fused posterior."""

    rank: int
    lat: float
    lon: float
    score: float
    """Posterior probability mass attributed to this candidate's cell."""
    label: str = ""
    country: str = ""
    admin1: str = ""
    nearest_place: str = ""
    distance_to_place_km: float | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    osm_matches: list[dict[str, Any]] = Field(default_factory=list)


class ImageFacts(BaseModel):
    """Basic physical facts about the file under analysis."""

    path: str
    filename: str
    sha256: str
    bytes: int
    width: int
    height: int
    format: str
    mode: str


class FaceFinding(BaseModel):
    count: int = 0
    boxes: list[tuple[int, int, int, int]] = Field(default_factory=list)
    blurred_export: str | None = None
    detector: str = ""


class CaseReport(BaseModel):
    """The complete, serialisable result of analysing one image."""

    case_id: str
    created_utc: datetime
    image: ImageFacts
    evidence: list[Evidence] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    faces: FaceFinding = Field(default_factory=FaceFinding)
    offline_mode: bool = True
    analyzers_run: list[str] = Field(default_factory=list)
    analyzers_skipped: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    posterior_png: str | None = None
    """Relative path to the rendered posterior heatmap overlay, if produced."""
    top_countries: list[tuple[str, float]] = Field(default_factory=list)
    entropy_bits: float | None = None
    """Shannon entropy of the posterior. High entropy = the evidence did not
    meaningfully constrain the location; surfaced so the analyst is not
    misled by a confident-looking top candidate."""

    def evidence_by_id(self, eid: str) -> Evidence | None:
        return next((e for e in self.evidence if e.id == eid), None)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def new_case_id(image_path: Path) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = "".join(c for c in image_path.stem if c.isalnum() or c in "-_")[:32]
    return f"{stamp}_{stem or 'case'}"
