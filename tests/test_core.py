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
    assert fields == {"count", "boxes", "blurred_export", "detector",
                      "people_count"}, (
        f"FaceFinding gained unexpected fields: {fields}")
    # Every field must be geometry or a tally -- never a descriptor, embedding
    # or attribute that could identify or characterise a person.
    for name in ("count", "people_count"):
        assert FaceFinding.model_fields[name].annotation is int


# ---------------------------------------------------------------------------
# Packaging: frozen-app paths and the optional scene model
# ---------------------------------------------------------------------------

def test_app_support_dir_is_outside_the_bundle():
    """A .app is read-only, so nothing may be written next to the binary."""
    from geoloc.config import PKG_ROOT, app_support_dir

    support = app_support_dir()
    assert not str(support).startswith(str(PKG_ROOT))
    assert support.is_absolute()


def test_frozen_paths_redirect_to_app_support(monkeypatch):
    """Frozen builds must not default case output into the bundle."""
    import geoloc.config as cfg

    monkeypatch.setattr(cfg.sys, "frozen", True, raising=False)
    monkeypatch.setattr(cfg.sys, "_MEIPASS", "/tmp/meipass", raising=False)
    assert cfg.is_frozen()
    assert cfg.app_support_dir() in cfg.default_case_dir().parents
    assert cfg.app_support_dir() in cfg.default_model_cache().parents


def test_resolve_device_survives_a_broken_torch(monkeypatch):
    """The device probe backs the app's readiness check.

    Regression: inside the PyInstaller bundle `torch.backends` was not
    collected, so attribute access raised AttributeError, every request to
    /api/settings returned 500, and the app read as 'will not start'.
    """
    import builtins

    from geoloc.config import Settings

    real_import = builtins.__import__

    class Stub:  # torch with no `backends` attribute, as in the bad bundle
        pass

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return Stub()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert Settings(device="auto").resolve_device() == "cpu"


def test_model_status_shape():
    from geoloc import modelmgr

    s = modelmgr.status()
    for key in ("model", "revision", "installed", "torch_available",
                "cache_dir", "approx_bytes", "progress"):
        assert key in s, f"status() missing {key}"
    assert isinstance(s["installed"], bool)
    for key in ("state", "downloaded", "total", "percent"):
        assert key in s["progress"]


def test_model_revision_is_pinned():
    """Letting transformers resolve the revision fetched two weight formats,
    turning a 1.7 GB download into 3.2 GB on disk."""
    from geoloc import modelmgr

    assert len(modelmgr.MODEL_REVISION) == 40, "revision must be a full commit sha"
    assert "pytorch_model.bin" in modelmgr.ALLOW_PATTERNS
    # The repo also carries demo photographs that serve no runtime purpose.
    assert not any(p.endswith((".jpg", ".jpeg", ".png"))
                   for p in modelmgr.ALLOW_PATTERNS)


def test_model_install_blocked_while_offline(monkeypatch):
    from geoloc import modelmgr
    from geoloc.config import SETTINGS

    monkeypatch.setattr(SETTINGS, "offline", True)
    monkeypatch.setattr(modelmgr, "is_installed", lambda: False)
    result = modelmgr.start_download()
    assert result["ok"] is False
    assert "offline" in result["reason"].lower()


def test_scene_analyzer_reports_missing_model_as_evidence(tmp_path, monkeypatch):
    """A missing optional model is a finding, not a crash."""
    from PIL import Image

    from geoloc.analyzers import clip_scene

    img = tmp_path / "x.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(img)

    def boom(_path):
        raise clip_scene.SceneModelUnavailable("The scene model is not installed.")

    monkeypatch.setattr(clip_scene, "_encode_image", boom)
    evidence = clip_scene.analyze(img)
    assert len(evidence) == 1
    assert evidence[0].id == "clip.unavailable"
    assert evidence[0].constraints == [], "an absent model must not constrain location"


# ---------------------------------------------------------------------------
# Precision reporting
# ---------------------------------------------------------------------------

def test_credible_area_separates_a_point_from_a_continent(grid):
    """The headline honesty figure must actually discriminate.

    Regression: entropy alone did not. A country-level result scored 15.1
    bits against a 17.8-bit uniform world, which sounds constrained, so the
    warning never fired and a ranked list of unrelated cities was presented
    as though it were a set of leads.
    """
    from geoloc.geo.fusion import credible_region_km2, describe_precision

    uniform = grid.cell_area / grid.cell_area.sum()
    tight = np.exp(-((grid.lat2d - 45.8) ** 2 + (grid.lon2d - 16.0) ** 2) / 0.02)
    tight = tight * grid.cell_area
    tight /= tight.sum()

    area_uniform = credible_region_km2(uniform, grid)
    area_tight = credible_region_km2(tight, grid)
    assert area_tight < area_uniform / 1000
    assert describe_precision(area_uniform)[0] == "unconstrained"
    assert describe_precision(area_tight)[0] in {"pinpoint", "locality", "regional"}


def test_country_only_evidence_is_reported_as_country_level(grid):
    """A single country constraint must not read as a located answer."""
    from geoloc.geo.fusion import credible_region_km2, describe_precision, fuse

    ev = Evidence(
        id="clip.country", analyzer="scene", title="country",
        confidence=Confidence.MEDIUM,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"AU": 1.0}, floor=0.10,
                                   confidence=Confidence.MEDIUM)])
    post, g = fuse([ev], grid=grid)
    band, _ = describe_precision(credible_region_km2(post, g))
    assert band in {"country", "unconstrained"}, (
        f"country-only evidence reported as {band}")


def test_precision_bands_are_ordered():
    from geoloc.geo.fusion import describe_precision

    areas = [1, 500, 50_000, 1_000_000, 50_000_000]
    bands = [describe_precision(a)[0] for a in areas]
    assert bands == ["pinpoint", "locality", "regional", "country",
                     "unconstrained"]


# ---------------------------------------------------------------------------
# Text geocoding
# ---------------------------------------------------------------------------

def test_generic_signage_is_not_geocoded():
    """Querying a gazetteer for "LIBRARY" returns a library, just not the
    right one. Generic words must never become a location constraint."""
    from geoloc.geo.textgeo import _informative

    for generic in ("LIBRARY", "BUSINESS", "ENTRANCE", "CAR PARK", "Hotel",
                    "RY", "SINES", "12345", "no"):
        assert not _informative(generic), f"{generic!r} should be rejected"

    for specific in ("PEKARNA DUBRAVICA", "Bond University", "Ilica 5",
                     "Restaurante La Habana"):
        assert _informative(specific), f"{specific!r} should be accepted"


def test_candidate_queries_prefers_confident_and_joins_lines():
    from geoloc.geo.textgeo import candidate_queries

    ev = Evidence(
        id="ocr.text", analyzer="text", title="text",
        raw={"blocks": [
            {"text": "DUBRAVICA", "confidence": 0.9},
            {"text": "PEKARNA", "confidence": 0.95},
            {"text": "LIBRARY", "confidence": 0.99},
        ]})
    queries = candidate_queries([ev])
    assert "LIBRARY" not in queries, "generic token leaked into queries"
    # A shopfront split across lines should also be tried as one phrase.
    assert any(" " in q for q in queries), f"no joined phrase in {queries}"


def test_geocoding_is_blocked_offline(monkeypatch):
    from geoloc.config import SETTINGS
    from geoloc.geo import textgeo

    monkeypatch.setattr(SETTINGS, "offline", True)
    ev = Evidence(id="ocr.text", analyzer="text", title="t",
                  raw={"blocks": [{"text": "PEKARNA DUBRAVICA", "confidence": 0.9}]})
    assert textgeo.geocode_evidence([ev]) == []


# ---------------------------------------------------------------------------
# OCR tiling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("sys").platform != "darwin", reason="Apple Vision OCR is macOS-only")
def test_tiled_ocr_reads_small_signage(tmp_path):
    """Small text in a wide frame must not be silently dropped.

    Regression: Vision returns nothing for signage below roughly 1/32 of the
    image height, so a campus photo with legible "LIBRARY" lettering scored
    as "no legible text detected" -- discarding the most valuable cue the
    tool has.
    """
    from PIL import Image, ImageDraw, ImageFont

    from geoloc.analyzers.text_ocr import run_ocr

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    except OSError:
        pytest.skip("system font unavailable")

    # 1600x1200 with ~26px lettering: about 2% of image height, under the
    # threshold at which a whole-image pass reliably sees it.
    img = Image.new("RGB", (1600, 1200), (205, 200, 190))
    d = ImageDraw.Draw(img)
    d.rectangle([180, 520, 700, 580], fill=(238, 236, 230))
    d.text((200, 536), "PEKARNA DUBRAVICA", fill=(20, 20, 20), font=font)
    path = tmp_path / "wide.png"
    img.save(path)

    blocks, engine = run_ocr(path)
    assert engine == "apple-vision"
    text = " ".join(b["text"] for b in blocks).upper()
    assert "DUBRAVICA" in text, f"tiled OCR missed the signage; got {text!r}"


# ---------------------------------------------------------------------------
# Facility gazetteer
# ---------------------------------------------------------------------------

def test_facility_inference_from_signage():
    """Generic words are useless for geocoding but strong facility evidence."""
    from geoloc.geo.facility import infer_facilities

    uni = infer_facilities(["LIBRARY", "BUSINESS", "FACULTY OF LAW"])
    assert uni and uni[0][0].key == "university"

    air = infer_facilities(["DEPARTURES", "GATE 14", "BAGGAGE RECLAIM"])
    assert air and air[0][0].key == "airport"

    # A shop name implies no particular facility type; it is geocoded instead.
    assert infer_facilities(["PEKARNA DUBRAVICA"]) == []


def test_sites_constraint_is_disjunctive(grid):
    """"One of these N places" must be an OR, not an AND.

    Rendering each site as a separate POINT constraint would multiply them
    together and drive the whole posterior to zero, since no cell is near all
    of them.
    """
    from geoloc.geo.fusion import render_constraint

    sites = [(-28.073, 153.417), (-33.917, 151.231), (-37.798, 144.961)]
    c = GeoConstraint(kind=ConstraintKind.SITES, sites=sites,
                      site_radius_km=3.0, floor=1e-4,
                      confidence=Confidence.MEDIUM)
    field = render_constraint(c, grid)
    for lat, lon in sites:
        r, col = grid.cell_of(lat, lon)
        assert field[r, col] > field.min(), "a listed site was not favoured"
    # Somewhere far from every site must be strongly disfavoured.
    r, col = grid.cell_of(20.0, 0.0)
    assert field[r, col] < field.max() - 5


def test_sites_constraint_narrows_the_credible_area(grid):
    """A site list must materially shrink the search area."""
    from geoloc.geo.fusion import credible_region_km2, fuse

    country = Evidence(
        id="clip.country", analyzer="scene", title="country",
        confidence=Confidence.MEDIUM,
        constraints=[GeoConstraint(kind=ConstraintKind.COUNTRIES,
                                   country_scores={"AU": 1.0}, floor=0.10,
                                   confidence=Confidence.MEDIUM)])
    post_before, g = fuse([country], grid=grid)

    sites = Evidence(
        id="facility.university.AU", analyzer="facility", title="sites",
        confidence=Confidence.MEDIUM,
        constraints=[GeoConstraint(
            kind=ConstraintKind.SITES,
            sites=[(-28.073, 153.417), (-27.5, 153.0), (-33.9, 151.2)],
            site_radius_km=2.0, floor=1e-4, confidence=Confidence.MEDIUM)])
    post_after, g = fuse([country, sites], grid=grid)

    assert (credible_region_km2(post_after, g)
            < credible_region_km2(post_before, g) / 10)


def test_candidates_are_named_by_site_not_cell(grid):
    """A gazetteer answer must report the place, not the grid cell.

    "Bond University (Robina, AU)" is the useful output; "the cell centred on
    -28.25, 153.25" is not, and hides that a named match was even found.
    """
    from geoloc.geo.fusion import extract_candidates, fuse

    ev = Evidence(
        id="facility.university.AU", analyzer="facility", title="sites",
        confidence=Confidence.MEDIUM,
        raw={"sites": [{"name": "Bond University", "lat": -28.0730,
                        "lon": 153.4165, "osm": "https://example.invalid/1"}]},
        constraints=[GeoConstraint(
            kind=ConstraintKind.SITES, sites=[(-28.0730, 153.4165)],
            site_radius_km=2.0, floor=1e-5, confidence=Confidence.MEDIUM)])
    post, g = fuse([ev], grid=grid)
    cands = extract_candidates(post, g, n=3, evidence=[ev])
    assert cands, "site constraint produced no candidate"
    assert "Bond University" in cands[0].label, cands[0].label
    # Exact site coordinates, not the cell centre.
    assert abs(cands[0].lat - (-28.0730)) < 1e-6
    assert abs(cands[0].lon - 153.4165) < 1e-6


# ---------------------------------------------------------------------------
# Analyst brief
# ---------------------------------------------------------------------------

def _bare_report(**kw):
    from geoloc.models import CaseReport, ImageFacts

    base = {
        "case_id": "t",
        "created_utc": datetime(2024, 1, 1, tzinfo=UTC),
        "image": ImageFacts(path="/x.jpg", filename="x.jpg", sha256="0" * 64,
                            bytes=1, width=10, height=10, format="JPEG",
                            mode="RGB"),
    }
    base.update(kw)
    return CaseReport(**base)


def test_brief_refuses_to_localise_a_country_level_result():
    """The headline must not name a place the evidence does not support."""
    from geoloc.models import Candidate
    from geoloc.summary import build_summary

    rep = _bare_report(
        precision_band="country", credible_area_km2=5_000_000.0,
        top_countries=[("AU", 0.9)],
        candidates=[Candidate(rank=1, lat=-19.3, lon=146.8, score=0.007,
                              label="James Cook University (Kirwan, AU)",
                              country="AU")])
    sm = build_summary(rep)
    assert "not evidence-backed" in sm["assessment"].lower() or \
           "not localised" in sm["assessment"].lower()
    assert "James Cook" not in sm["location"], (
        "a country-level result presented a specific site as the location")
    assert "indicative only" in sm["coordinates"]


def test_brief_localises_when_the_evidence_supports_it():
    from geoloc.models import Candidate
    from geoloc.summary import build_summary

    rep = _bare_report(
        precision_band="locality", credible_area_km2=900.0,
        top_countries=[("HR", 0.98)],
        candidates=[Candidate(rank=1, lat=45.8125, lon=15.975, score=0.53,
                              label="Centar, HR", country="HR")])
    sm = build_summary(rep)
    assert "Centar" in sm["location"]
    assert "indicative only" not in sm["coordinates"]


def test_scene_readings_are_attributed_not_asserted():
    """Scene-model output must never read as observed fact.

    Regression: the brief described a Gold Coast university campus as "a
    beach with sub saharan african housing and hot desert vegetation" -- four
    confident assertions, all wrong, from softmax scores above 0.55.
    """
    from geoloc.summary import describe_scene

    rep = _bare_report(evidence=[
        Evidence(id="clip.biome", analyzer="scene", title="b",
                 raw={"ranked": [["hot desert", 0.73]]}),
        Evidence(id="clip.architecture", analyzer="scene", title="a",
                 raw={"ranked": [["sub saharan african housing", 0.57]]}),
    ])
    text = describe_scene(rep)
    assert "Scene model readings" in text
    assert "unreliable" in text.lower()
    assert not text.startswith("Appears to show"), (
        "model guesses are being asserted as description")


def test_brief_time_of_day_prefers_exif_then_solar_then_says_nothing():
    from geoloc.summary import estimate_time_of_day

    empty = _bare_report()
    assert estimate_time_of_day(empty, shadow_azimuth=None,
                                capture_date=None)["method"] == "none"

    with_exif = _bare_report(evidence=[
        Evidence(id="meta.timestamp", analyzer="metadata", title="t",
                 raw={"datetime_original": "2024:06:21 14:30:00",
                      "offset": "+02:00"})])
    got = estimate_time_of_day(with_exif, shadow_azimuth=None, capture_date=None)
    assert got["method"] == "exif" and "2024:06:21" in got["local_time"]


def test_solar_solver_recovers_time_of_day_from_a_shadow():
    """Given a location and a shadow bearing, the sun dates the photograph.

    Ground truth: 09:30 AEST at Bond University on 2024-08-15 is 23:30 UTC on
    the 14th; the bearing is computed from the solver and fed back in.
    """
    from geoloc.analyzers.solar import local_solar_time, solar_position, to_julian_day

    lat, lon = -28.073, 153.417
    truth = datetime(2024, 8, 14, 23, 30, tzinfo=UTC)
    _el, az = solar_position(to_julian_day(truth), np.array([lat]), np.array([lon]))
    shadow = (float(az[0]) + 180.0) % 360.0

    solved = local_solar_time(lat, lon, datetime(2024, 8, 14, tzinfo=UTC), shadow)
    assert solved is not None
    instant, elevation, err = solved
    assert abs((instant - truth).total_seconds()) <= 600, instant
    assert err < 1.0
    assert elevation > 0


# ---------------------------------------------------------------------------
# Vehicle registration identifiers
# ---------------------------------------------------------------------------

def test_uic_checksum_rejects_lookalikes():
    """A run of digits must not become a HIGH-confidence country claim.

    UIC numbers carry a Luhn check digit; without verifying it, a phone
    number or a price would inject a country constraint stronger than
    anything the scene model produces.
    """
    from geoloc.analyzers.text_ocr import parse_uic, uic_check_digit

    valid = "93870029001" + str(uic_check_digit("93870029001"))
    assert parse_uic(valid) == "FR"
    assert parse_uic("93 87 0029") == "FR"          # partial, valid type+country

    assert parse_uic("93 87 0029 001-2") is None    # wrong check digit
    assert parse_uic("+33 1 45 67 89 01") is None   # French phone number
    assert parse_uic("19.99 EUR 2024 0012") is None
    assert parse_uic("12 34 5678 9012") is None     # country code 34 unassigned


def test_uic_is_not_assembled_across_ocr_blocks():
    """Vehicle numbers are single painted strings.

    Regression: the pattern allowed \\s, so a neighbouring block's stray digit
    was glued on ("4\\n93 87 0029 001-9"), pushing the run past the length
    check and silently discarding a valid number.
    """
    from geoloc.analyzers.text_ocr import UIC_RE

    joined = "PLATFORM 4\n93 87 0029 001-9"
    for m in UIC_RE.finditer(joined):
        assert "\n" not in m.group(0), f"matched across a line break: {m.group(0)!r}"


def test_aircraft_registration_prefixes():
    from geoloc.analyzers.text_ocr import AIRCRAFT_RE
    from geoloc.data.reference import AIRCRAFT_PREFIX_COUNTRY

    for reg, iso2 in (("F-GKXA", "FR"), ("G-EUPT", "GB"), ("D-AIMA", "DE")):
        assert AIRCRAFT_RE.search(reg), reg
        prefix = next(p for p in sorted(AIRCRAFT_PREFIX_COUNTRY, key=len,
                                        reverse=True) if reg.startswith(p))
        assert AIRCRAFT_PREFIX_COUNTRY[prefix] == iso2


def test_facility_keywords_are_never_geocoded():
    """The generic list and the facility list must not drift apart.

    Regression: "platform" was a facility keyword but not a generic token, so
    "PLATFORM 4" was geocoded, matched a business in Australia, and moved an
    entire French case to Victoria.
    """
    from geoloc.geo.facility import FACILITIES
    from geoloc.geo.textgeo import GENERIC_TOKENS, _informative

    for fac in FACILITIES:
        for kw in fac.keywords:
            for word in kw.split():
                if len(word) > 1:
                    assert word in GENERIC_TOKENS, (
                        f"facility keyword {word!r} is missing from the "
                        "generic-token list and could be geocoded")

    assert not _informative("PLATFORM 4")
    assert not _informative("DEPARTURES")
    # A genuine proper noun must still get through.
    assert _informative("PEKARNA DUBRAVICA")


# ---------------------------------------------------------------------------
# Visual place recognition: abstention
# ---------------------------------------------------------------------------

def _hit(site, score, name="", lat=0.0, lon=0.0, title="t.jpg"):
    return (score, {"site_id": site, "site_name": name or site,
                    "site_lat": lat, "site_lon": lon, "title": title})


class _FakeIndex:
    def __init__(self, hits):
        self._hits = sorted(hits, key=lambda h: -h[0])
        self.records = [h[1] for h in self._hits]

    def __len__(self):
        return len(self._hits)

    def search(self, query, top_k=25):
        return self._hits[:top_k]


def test_vpr_rejects_when_nothing_matches():
    """An index always has a nearest neighbour; returning it regardless is
    how automated geolocation produces confident, precise, wrong answers.

    The scores here are the ones actually measured when MegaLoc was queried
    with a station photograph whose scene was NOT in the corpus: a best of
    0.16 with the field close behind. Reporting that as an identification
    would be fabrication.
    """
    from geoloc.vpr.match import match

    idx = _FakeIndex([_hit("a", 0.16), _hit("b", 0.11), _hit("c", 0.08)])
    assert not match(idx, np.zeros(4)).accepted

    # And a corpus containing nothing even loosely similar.
    faint = _FakeIndex([_hit("a", 0.09), _hit("b", 0.02)])
    result = match(faint, np.zeros(4))
    assert not result.accepted
    assert "floor" in result.reason


def test_vpr_rejects_a_tie_between_sites():
    """Two sites scoring alike means the descriptor is responding to
    something they share, not to either of them.

    Calibration put this band squarely where identifications go wrong:
    accepting a margin of 0.01 drops precision to 84%, while requiring 0.05
    raises it to 96%.
    """
    from geoloc.vpr.match import match

    idx = _FakeIndex([_hit("a", 0.71), _hit("b", 0.70), _hit("c", 0.4)])
    result = match(idx, np.zeros(4))
    assert not result.accepted
    assert "separation" in result.reason


def test_vpr_absolute_score_is_only_a_floor():
    """Relative separation carries the signal, not absolute similarity.

    Regression: the first implementation accepted on absolute score alone.
    Held-out calibration showed that is the weakest of four discriminators
    (AUC 0.78 against 0.87 for the ratio), and a floor set from the control
    distribution would have rejected 68% of genuine matches. A modest
    absolute score with clear separation must therefore be accepted.
    """
    from geoloc.vpr.match import match

    idx = _FakeIndex([_hit("a", 0.34, "Real Place", 43.3, 5.4),
                      _hit("a", 0.30, "Real Place", 43.3, 5.4),
                      _hit("b", 0.15, "Elsewhere", 48.8, 2.3)])
    result = match(idx, np.zeros(4))
    assert result.accepted, result.reason
    assert result.matches[0].site_name == "Real Place"


def test_vpr_accepts_a_clear_well_supported_match():
    from geoloc.vpr.match import match

    idx = _FakeIndex([
        _hit("gare", 0.82, "Marseille-Saint-Charles", 43.303, 5.381),
        _hit("gare", 0.77, "Marseille-Saint-Charles", 43.303, 5.381),
        _hit("other", 0.51, "Somewhere Else", 48.8, 2.3),
    ])
    result = match(idx, np.zeros(4))
    assert result.accepted, result.reason
    best = result.matches[0]
    assert best.site_name == "Marseille-Saint-Charles"
    assert best.support == 2, "supporting images should be counted"
    assert abs(best.lat - 43.303) < 1e-6


def test_vpr_aggregates_images_to_sites_keeping_the_best_score():
    from geoloc.vpr.match import aggregate

    sites = aggregate([_hit("x", 0.5), _hit("x", 0.9), _hit("y", 0.7)])
    assert [s.site_id for s in sites] == ["x", "y"]
    assert sites[0].best_score == 0.9
    assert sites[0].support == 2


def test_vpr_thresholds_are_documented_constants():
    """The thresholds decide whether this tool fabricates. They must be
    explicit and adjustable, not buried literals."""
    from geoloc.vpr import match as m

    assert 0.0 < m.MIN_SIMILARITY < 1.0
    assert 0.0 < m.MIN_MARGIN < 1.0
    assert m.match.__doc__ is not None or m.__doc__ is not None
    assert "abstention" in m.__doc__.lower()


def test_vpr_match_labels_the_candidate_by_site_name():
    """A visual match must report the place it found, not the nearest town.

    "Marseille Saint-Charles" is the finding; "Marseille 03" is the postcode
    district that happens to contain it and tells an analyst nothing about
    what was matched.
    """
    from geoloc.geo.fusion import extract_candidates, fuse
    from geoloc.geo.grid import world_grid

    grid = world_grid(1.0)
    ev = Evidence(
        id="vpr.match.fr-stations", analyzer="vpr", title="Visual match",
        confidence=Confidence.HIGH,
        raw={"sites": [{"name": "Marseille Saint-Charles",
                        "lat": 43.3033, "lon": 5.3812}]},
        constraints=[GeoConstraint(kind=ConstraintKind.POINT, lat=43.3033,
                                   lon=5.3812, radius_km=0.4,
                                   confidence=Confidence.HIGH)])
    post, g = fuse([ev], grid=grid)
    cands = extract_candidates(post, g, n=2, evidence=[ev])
    assert cands and "Marseille Saint-Charles" in cands[0].label
    assert abs(cands[0].lat - 43.3033) < 1e-6


def test_corpus_prefers_curated_categories_over_a_radius():
    """Corpus quality is the accuracy bottleneck, so harvesting must lead
    with categories.

    A 400 m radius around a station returns a vending machine, a bin
    collection and a commemorative plaque -- all genuinely nearby, none
    useful for recognising the place. Commons categories are curated as
    images *of* a subject, so they are queried first and the radius only
    fills out coverage.
    """
    import inspect

    from geoloc.vpr import corpus

    src = inspect.getsource(corpus.harvest_site)
    cat_at = src.index("find_category")
    geo_at = src.index("discover_files")
    assert cat_at < geo_at, "category lookup must precede the radius search"
    assert hasattr(corpus, "category_files")


def test_category_traversal_descends_into_subcategories():
    """A station's own category is often near-empty while its subcategories
    hold everything, so one level of descent is required."""
    import inspect

    from geoloc.vpr import corpus

    sig = inspect.signature(corpus.category_files)
    assert sig.parameters["depth"].default >= 1
