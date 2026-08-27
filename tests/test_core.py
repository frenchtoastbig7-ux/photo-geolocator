"""Regression tests for the analytical core.

The astronomy and fusion tests use published/derivable ground truth rather
than golden values captured from this implementation, so they would catch a
genuine regression rather than merely freezing current behaviour.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from geoloc.analyzers.solar import (
    elevation_from_shadow_ratio,
    shadow_heatmap,
    solar_position,
    to_julian_day,
)
from geoloc.analyzers.text_ocr import classify_scripts
from geoloc.geo.fusion import (
    detect_contradictions,
    extract_candidates,
    fuse,
    posterior_entropy_bits,
    render_constraint,
    top_countries,
    uniform_entropy_bits,
)
from geoloc.geo.grid import haversine_km, world_grid
from geoloc.models import Confidence, ConstraintKind, Evidence, GeoConstraint

STEP = 1.0  # coarse grid keeps the suite fast


@pytest.fixture(scope="module")
def grid():
    return world_grid(STEP)


# ---------------------------------------------------------------------------
# Astronomy
# ---------------------------------------------------------------------------

def test_julian_day_j2000():
    """J2000.0 epoch is JD 2451545.0 by definition."""
    jd = to_julian_day(datetime(2000, 1, 1, 12, 0, tzinfo=UTC))
    assert abs(jd - 2451545.0) < 1e-6


def test_julian_day_requires_tzinfo():
    with pytest.raises(ValueError):
        to_julian_day(datetime(2024, 1, 1, 12, 0))


@pytest.mark.parametrize("lat,lon,when,exp_el,exp_az", [
    # Greenwich, summer solstice noon UTC: sun due south, ~62 deg up.
    (51.4779, 0.0, datetime(2024, 6, 21, 12, 0, tzinfo=UTC), 61.9, 180.0),
    # Same place, winter solstice noon: same azimuth, ~15 deg up.
    (51.4779, 0.0, datetime(2024, 12, 21, 12, 0, tzinfo=UTC), 15.1, 180.0),
])
def test_solar_position_known_values(lat, lon, when, exp_el, exp_az):
    el, az = solar_position(to_julian_day(when), np.array([lat]), np.array([lon]))
    assert abs(float(el[0]) - exp_el) < 1.5, f"elevation {el[0]:.2f} vs {exp_el}"
    assert abs((float(az[0]) - exp_az + 180) % 360 - 180) < 2.0


def test_sun_below_horizon_at_antipode():
    """When it is local noon at Greenwich the sun is down near the antimeridian."""
    jd = to_julian_day(datetime(2024, 6, 21, 12, 0, tzinfo=UTC))
    el, _ = solar_position(jd, np.array([0.0]), np.array([180.0]))
    assert float(el[0]) < 0


def test_equator_equinox_near_zenith():
    jd = to_julian_day(datetime(2024, 3, 20, 12, 0, tzinfo=UTC))
    el, _ = solar_position(jd, np.array([0.0]), np.array([0.0]))
    assert float(el[0]) > 85.0


def test_shadow_ratio_to_elevation():
    assert elevation_from_shadow_ratio(1, 1) == pytest.approx(45.0)
    assert elevation_from_shadow_ratio(1, math.sqrt(3)) == pytest.approx(30.0, abs=1e-6)
    with pytest.raises(ValueError):
        elevation_from_shadow_ratio(1, 0)


def test_shadow_heatmap_recovers_known_location(grid):
    """A shadow bearing plus a UTC instant should localise the observer.

    Ground truth: Greenwich at 2024-06-21 12:00 UTC has the sun at azimuth
    ~179 deg / elevation ~62 deg, so the shadow points to ~359 deg.
    """
    field, mode = shadow_heatmap(
        grid, when_utc=datetime(2024, 6, 21, 12, 0, tzinfo=UTC),
        shadow_azimuth=359.0, sun_elevation=62.0)
    assert mode == "instant"
    r, c = np.unravel_index(int(np.argmax(field)), field.shape)
    err = haversine_km(float(grid.lat2d[r, c]), float(grid.lon2d[r, c]), 51.48, 0.0)
    assert err < 250, f"peak {err:.0f} km from Greenwich"


def test_date_only_mode_gives_no_longitude_information(grid):
    """Without a clock, shadows constrain latitude only.

    This is the property that stops the tool inventing a longitude it cannot
    know, so it is asserted directly: every column must be identical.
    """
    field, mode = shadow_heatmap(
        grid, date_only=datetime(2024, 6, 21, tzinfo=UTC),
        shadow_azimuth=359.0, sun_elevation=62.0)
    assert mode == "date-only"
    assert np.allclose(field, field[:, :1], atol=1e-12)


# ---------------------------------------------------------------------------
# Constraints and fusion
# ---------------------------------------------------------------------------

def test_point_constraint_survives_coarse_grid(grid):
    """A tight point constraint must not underflow away on a coarse grid.

    Regression: a 2 km radius sampled on ~110 km cells previously fell below
    the log floor at every cell, silently discarding an EXIF GPS tag.
    """
    c = GeoConstraint(kind=ConstraintKind.POINT, lat=45.8125, lon=15.975,
                      radius_km=2.0, confidence=Confidence.CERTAIN)
    field = render_constraint(c, grid)
    assert field.max() > field.min(), "point constraint collapsed to a constant"
    r, co = np.unravel_index(int(np.argmax(field)), field.shape)
    assert haversine_km(float(grid.lat2d[r, co]), float(grid.lon2d[r, co]),
                        45.8125, 15.975) < 120


def test_gps_evidence_yields_exact_coordinates(grid):
    """Reported coordinates come from the tag, not the cell centre."""
    ev = Evidence(
        id="meta.gps", analyzer="metadata", title="GPS",
        confidence=Confidence.CERTAIN,
        constraints=[GeoConstraint(kind=ConstraintKind.POINT, lat=45.8125,
                                   lon=15.975, radius_km=2.0,
                                   confidence=Confidence.CERTAIN)])
    post, g = fuse([ev], grid=grid)
    cands = extract_candidates(post, g, n=3, evidence=[ev])
    assert cands, "no candidate produced from a certain GPS tag"
    assert cands[0].lat == pytest.approx(45.8125)
    assert cands[0].lon == pytest.approx(15.975)


def test_country_constraint_concentrates_mass(grid):
    ev = Evidence(
        id="ocr.cctld", analyzer="text", title="ccTLD", confidence=Confidence.HIGH,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"HR": 1.0}, floor=0.05,
                                   confidence=Confidence.HIGH)])
    post, g = fuse([ev], grid=grid)
    ranked = top_countries(post, g, n=3)
    assert ranked[0][0] == "HR"
    assert ranked[0][1] > 0.5


def test_soft_constraints_never_zero_out_the_truth(grid):
    """A wrong soft cue must not make the correct answer impossible.

    Confidently-wrong automated geolocation is the main failure mode this
    tool is built to avoid, so the floor behaviour is asserted explicitly.
    """
    wrong = Evidence(
        id="clip.country", analyzer="scene", title="wrong guess",
        confidence=Confidence.MEDIUM,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"BR": 1.0}, floor=0.10,
                                   confidence=Confidence.MEDIUM)])
    post, g = fuse([wrong], grid=grid)
    hr = post[g.country_mask("HR")]
    assert hr.size and hr.sum() > 0, "a soft cue eliminated an entire country"


def test_contradictory_evidence_is_reported(grid):
    """A GPS tag disagreeing with scene content must be surfaced, not buried.

    A CERTAIN constraint legitimately dominates the posterior, so the
    conflict cannot be seen in the fused output; it has to be detected on
    the individual likelihood fields and reported.
    """
    gps = Evidence(
        id="meta.gps", analyzer="metadata", title="Embedded GPS coordinates",
        confidence=Confidence.CERTAIN,
        constraints=[GeoConstraint(kind=ConstraintKind.POINT, lat=45.81, lon=15.97,
                                   radius_km=2.0, confidence=Confidence.CERTAIN)])
    scene = Evidence(
        id="clip.country", analyzer="scene", title="Zero-shot country estimate",
        confidence=Confidence.HIGH,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"JP": 1.0}, floor=0.02,
                                   confidence=Confidence.HIGH)])
    warnings = detect_contradictions([gps, scene], grid)
    assert warnings, "conflicting GPS and scene evidence was not reported"
    assert "Contradictory evidence" in warnings[0]


def test_agreeing_evidence_is_not_flagged(grid):
    """Consistent evidence must not produce spurious contradiction warnings."""
    gps = Evidence(
        id="meta.gps", analyzer="metadata", title="Embedded GPS coordinates",
        confidence=Confidence.CERTAIN,
        constraints=[GeoConstraint(kind=ConstraintKind.POINT, lat=45.81, lon=15.97,
                                   radius_km=2.0, confidence=Confidence.CERTAIN)])
    tld = Evidence(
        id="ocr.cctld", analyzer="text", title="ccTLD", confidence=Confidence.HIGH,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"HR": 1.0}, floor=0.02,
                                   confidence=Confidence.HIGH)])
    assert detect_contradictions([gps, tld], grid) == []


def test_entropy_drops_as_evidence_accumulates(grid):
    uniform = uniform_entropy_bits(grid)
    weak, _ = fuse([], grid=grid, use_habitation_prior=False)
    strong_ev = Evidence(
        id="e", analyzer="t", title="t", confidence=Confidence.HIGH,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"HR": 1.0}, floor=0.01,
                                   confidence=Confidence.HIGH)])
    strong, _ = fuse([strong_ev], grid=grid, use_habitation_prior=False)
    assert posterior_entropy_bits(weak) == pytest.approx(uniform, abs=0.01)
    assert posterior_entropy_bits(strong) < posterior_entropy_bits(weak) - 3


def test_candidates_are_spatially_separated(grid):
    ev = Evidence(
        id="e", analyzer="t", title="t", confidence=Confidence.HIGH,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"FR": 1.0}, floor=0.01,
                                   confidence=Confidence.HIGH)])
    post, g = fuse([ev], grid=grid)
    cands = extract_candidates(post, g, n=5, min_separation_km=150, evidence=[ev])
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert haversine_km(a.lat, a.lon, b.lat, b.lon) > 100


def test_posterior_is_normalised(grid):
    ev = Evidence(id="e", analyzer="t", title="t",
                  constraints=[GeoConstraint(kind=ConstraintKind.LAT_BAND,
                                             lat_min=0, lat_max=20)])
    post, _ = fuse([ev], grid=grid)
    assert post.sum() == pytest.approx(1.0, abs=1e-9)
    assert (post >= 0).all()


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat,lon,iso2", [
    (51.5074, -0.1278, "GB"), (35.6762, 139.6503, "JP"),
    (-33.8688, 151.2093, "AU"), (40.7128, -74.0060, "US"),
])
def test_grid_country_labelling(grid, lat, lon, iso2):
    r, c = grid.cell_of(lat, lon)
    assert grid.countries[grid.country_idx[r, c]] == iso2


def test_habitation_prior_penalises_open_ocean(grid):
    prior = grid.habitation_log_prior()
    r_sea, c_sea = grid.cell_of(0.0, -140.0)     # mid-Pacific
    r_land, c_land = grid.cell_of(48.86, 2.35)   # Paris
    assert prior[r_sea, c_sea] < prior[r_land, c_land] - 2


def test_haversine_known_distance():
    # London -> Paris, ~344 km great circle.
    d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330 < d < 360


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def test_script_classification():
    counts = classify_scripts("Улица Ленина Hello 東京 مرحبا")
    assert counts["Cyrillic"] > 0 and counts["Latin"] > 0
    assert counts["Han"] > 0 and counts["Arabic"] > 0


# ---------------------------------------------------------------------------
# OCR (regression: Vision language-priority behaviour)
# ---------------------------------------------------------------------------

def test_pass_script_validation_rejects_transliteration_garbage():
    """A pass must not keep output that is not in the script it targets.

    Regression: running the Chinese-language pass over Cyrillic signage
    yields confident nonsense ("Улица Ленина" -> "ynuua JeHHa"). Without
    this filter that garbage reaches the evidence board and the ccTLD /
    phone-number miners.
    """
    from geoloc.analyzers.text_ocr import _accepts

    assert _accepts("東京都渋谷区", "Han")
    assert not _accepts("ynuua JeHHa 45", "Han")
    assert _accepts("Улица Ленина", "Cyrillic")
    assert not _accepts("Улица Ленина", "Latin")
    assert _accepts("PEKARNA", "Latin")
    assert _accepts("45", "Latin"), "bare street numbers must survive"


def test_ocr_pass_groups_are_single_script():
    """Each pass leads with the language whose script it targets.

    Vision uses `recognitionLanguages` as a priority list, so a pass whose
    first entry is in the wrong script silently returns nothing.
    """
    from geoloc.analyzers.text_ocr import _PASS_ACCEPTS, OCR_PASSES

    assert set(OCR_PASSES) == set(_PASS_ACCEPTS)
    assert OCR_PASSES["Hiragana"][0].startswith("ja")
    assert OCR_PASSES["Cyrillic"][0].startswith("ru")
    assert OCR_PASSES["Arabic"][0].startswith("ar")
    assert OCR_PASSES["Latin"][0].startswith("en")


@pytest.mark.skipif(
    __import__("sys").platform != "darwin", reason="Apple Vision OCR is macOS-only")
def test_multipass_ocr_reads_multiple_scripts(tmp_path):
    """End-to-end: a card mixing four scripts must yield all four."""
    from PIL import Image, ImageDraw, ImageFont

    from geoloc.analyzers.text_ocr import classify_scripts, run_ocr

    try:
        latin = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 48)
        cjk = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 48)
        uni = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 48)
    except OSError:
        pytest.skip("system fonts unavailable")

    img = Image.new("RGB", (1100, 460), (250, 250, 246))
    d = ImageDraw.Draw(img)
    d.text((60, 40), "PEKARNA www.test.hr", fill=(10, 10, 10), font=latin)
    d.text((60, 150), "Улица Ленина 45", fill=(10, 10, 10), font=uni)
    d.text((60, 265), "東京都渋谷区", fill=(10, 10, 10), font=cjk)
    path = tmp_path / "multiscript.png"
    img.save(path)

    blocks, engine = run_ocr(path)
    assert engine == "apple-vision"
    text = " ".join(b["text"] for b in blocks)
    scripts = classify_scripts(text)
    assert scripts.get("Latin", 0) > 0, f"Latin missing from {text!r}"
    assert scripts.get("Cyrillic", 0) > 0, f"Cyrillic missing from {text!r}"
    assert scripts.get("Han", 0) > 0, f"Han missing from {text!r}"


# ---------------------------------------------------------------------------
# transformers compatibility
# ---------------------------------------------------------------------------

def test_clip_feature_extraction_handles_both_api_shapes():
    """`get_*_features` returns a tensor in transformers 4.x, an object in 5.x.

    Regression: 5.x returns `BaseModelOutputWithPooling`, and calling
    `.norm()` on it raised AttributeError, silently disabling the entire
    scene analyzer.
    """
    from geoloc.analyzers.clip_scene import _features

    class Pooled:            # transformers 5.x shape
        pooler_output = "PROJECTED"
        last_hidden_state = "HIDDEN"

    plain = np.zeros((1, 4))  # transformers 4.x shape: a bare tensor
    assert _features(Pooled()) == "PROJECTED"
    assert _features(plain) is plain


# ---------------------------------------------------------------------------
# Face redaction guardrail
# ---------------------------------------------------------------------------

def test_vision_box_origin_inversion():
    """Vision's bottom-left origin must map to a top-left pixel box.

    Getting this backwards blurs a mirrored region and leaves the actual
    face untouched -- a redaction failure that looks like it worked.
    """
    from geoloc.analyzers.faces import vision_box_to_pixels

    # A box hugging the TOP of the image: Vision y is high (origin at bottom).
    assert vision_box_to_pixels(0.0, 0.8, 0.5, 0.2, 1000, 1000) == (0, 0, 500, 200)
    # A box hugging the BOTTOM: Vision y is 0.
    assert vision_box_to_pixels(0.0, 0.0, 0.5, 0.2, 1000, 1000) == (0, 800, 500, 200)
    # Dead centre stays centred under the flip.
    assert vision_box_to_pixels(0.4, 0.4, 0.2, 0.2, 1000, 1000) == (400, 400, 200, 200)


def test_redaction_destroys_pixels_in_the_box(tmp_path):
    """Blurring must actually be irreversible over the whole expanded region."""
    import cv2
    from PIL import Image

    from geoloc.analyzers.faces import write_redacted
    from geoloc.models import FaceFinding

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    src = tmp_path / "src.png"
    Image.fromarray(noise).save(src)

    finding = FaceFinding(count=1, boxes=[(150, 150, 100, 100)], detector="test")
    dest = tmp_path / "redacted.png"
    assert write_redacted(src, finding, dest) is not None

    out = cv2.imread(str(dest))
    orig = cv2.imread(str(src))
    box = out[150:250, 150:250].astype(float)
    orig_box = orig[150:250, 150:250].astype(float)
    # High-frequency detail inside the box must be largely gone.
    assert box.std() < orig_box.std() * 0.5, "redacted region retains detail"
    # A far corner must be untouched.
    assert np.array_equal(out[0:20, 0:20], orig[0:20, 0:20])


def test_face_detection_returns_boxes_only():
    """The guardrail must expose no identity signal, only geometry."""
    from geoloc.models import FaceFinding

    fields = set(FaceFinding.model_fields)
    assert fields == {"count", "boxes", "blurred_export", "detector"}, (
        f"FaceFinding gained unexpected fields: {fields}")
