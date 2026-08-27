# Photo Geolocation Workbench

[![CI](https://github.com/frenchtoastbig7-ux/photo-geolocator/actions/workflows/ci.yml/badge.svg)](https://github.com/frenchtoastbig7-ux/photo-geolocator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

A local single-image geolocation tool for OSINT practitioners. Given one
photograph, it extracts every geographic cue it can, fuses them into a
probability distribution over the globe, and hands back ranked candidate
locations with the reasoning that produced them.

It runs entirely on your own hardware. There is no hosted version and no
account. **The image never leaves your machine.**

## Scope

This tool geolocates **scenes** — terrain, signage, architecture, sun angle,
file metadata. It performs no facial recognition and no identity inference,
and that boundary is enforced in code rather than left to convention.

It is built for practitioners working under some authority that permits the
work: investigators, researchers, journalists verifying imagery, CTF and
training use. Geolocating photographs engages legal and ethical obligations
that vary by jurisdiction and by the mandate you hold. Those are yours to
satisfy. There is deliberately no hosted deployment, because a public
one-click version of this invites uses the design is meant to discourage.

---

## Why it is built this way

Automated geolocation fails in a specific and dangerous way: it produces a
confident, precise, wrong answer. A tool that says "Bucharest, 94%" when the
photo is from Sofia is worse than one that says nothing, because the number
does the analyst's thinking for them.

Three design choices push against that:

**Evidence is fused, not chained.** Each cue becomes a weighted likelihood
field over a global grid. Nothing is a filter, so no single mistaken cue can
eliminate the true location — every soft constraint keeps a non-zero floor.

**Uncertainty is reported as loudly as the answer.** Every case carries the
posterior's Shannon entropy. When the evidence barely constrains anything,
the tool says so in plain terms instead of presenting a top-ranked candidate
drawn from noise.

**Contradictions are surfaced.** A GPS tag disagreeing with the scene is a
finding — a forged tag, a photo of a screen, a misattributed file — so
conflicting strong evidence is detected and reported rather than being
silently steamrollered by whichever cue carries more weight.

The output is a set of hypotheses to investigate. It is not an identification,
and nothing here substitutes for corroboration.

---

## Install

Requires Python 3.11+. On Apple Silicon the scene model runs on the GPU via
Metal automatically.

```bash
git clone https://github.com/frenchtoastbig7-ux/photo-geolocator.git
cd photo-geolocator
uv venv --python 3.12
uv pip install -e ".[ml,ocr]"
```

`ocr` is macOS-only (Apple Vision, on-device). `ml` pulls PyTorch and is
needed only for the scene model — everything else works without it. Add
`dev` for pytest and ruff.

Optional but recommended — richer metadata coverage (XMP, IPTC, MakerNotes):

```bash
brew install exiftool
```

Fetch the scene model once (~1.7 GB). After this the tool can run air-gapped:

```bash
geoloc prefetch-models
```

---

## Use

### Web workbench

```bash
geoloc serve
```

Opens on `http://127.0.0.1:8711`. Drag in an image, add anything you have
determined yourself, and analyse. Candidates appear on a map over a posterior
heatmap, beside the evidence board that produced them.

### Command line

```bash
geoloc analyze photo.jpg
```

Fully offline, with a shadow measurement and a known capture instant:

```bash
geoloc analyze photo.jpg --offline \
  --utc 2024-06-21T14:30:00 \
  --shadow-azimuth 135 --object-height 1.8 --shadow-length 3.1 \
  --driving-side right --country HR --country SI
```

Every run writes a case directory containing the source image, the posterior
as both `.npy` and a rendered PNG, `case.json`, and a self-contained
`report.html` that opens with no server and no network.

---

## What it analyses

| Analyzer | What it extracts | Typical strength |
|---|---|---|
| **Metadata** | EXIF/XMP GPS, capture time, UTC offset → longitude band, device, editing software, stripping indicators | Decisive when present, rare after any upload |
| **Text (OCR)** | On-device multilingual OCR; script → country, ccTLDs, dialling prefixes, diacritics | Often the single most decisive cue |
| **Scene (StreetCLIP)** | Zero-shot country, biome → latitude band, architecture, season, utilities, setting | Broad but overconfident; weighted low |
| **Solar geometry** | Shadow bearing + elevation vs. astronomical sun position | Rigorous; the strongest non-metadata technique |
| **Analyst input** | Driving side, country shortlist, exclusions, search area, notes | Outranks every model in the pipeline |
| **Faces** | Presence detection and redaction — a guardrail, not a capability | — |

### Solar geolocation, and its one sharp edge

With a **full UTC instant**, azimuth and elevation together pin latitude *and*
longitude, often to a band tens of kilometres wide.

With a **date but no time of day**, marginalising over the unknown hour
destroys the longitude information completely. The result is a
latitude-only constraint, and the tool reports it as exactly that. A
"date-only" result that appears to name a longitude would be an artefact, and
there is a test asserting it cannot happen.

Bearings are measured clockwise from **true** north, along the shadow away
from the object. Correct for magnetic declination before entering one.

---

## Network posture

Local-first, with opt-in outbound.

**Never leaves the machine:** the image, any crop of it, any embedding or
hash of it. There is no code path that transmits pixel data.

**Leaves only when you ask:** place names you choose to geocode and OSM tag
filters you select, sent to Nominatim/Overpass. Map tiles when the basemap is
on. Model weights, once, on first fetch.

Offline mode (the header toggle, or `--offline`) blocks all of it, including
tiles. Analysis is unaffected: everything except OSM refinement is local
regardless.

The distinction that matters: someone observing your traffic sees that you
searched for a clock tower near Zagreb. They do not see the photograph, and
cannot recover it.

Bind to loopback. There is no authentication because there is no intended
remote user.

---

## On images of people

The tool geolocates **scenes**. It performs no facial recognition, computes
no face embeddings, and does no identity inference — face detection returns
bounding boxes only, so that faces can be flagged and blurred in exported
report imagery.

That boundary is deliberate. Scene geolocation and person-tracking are
different activities with different authorisation requirements, and the
second should not arrive as a side effect of building the first. When faces
are detected the case carries a warning to that effect.

Geolocating imagery of identifiable people engages legal and ethical
obligations that vary by jurisdiction and by the authority you are operating
under. Those are yours to satisfy; the tool will not assess them for you.

---

## Accuracy, honestly

- **EXIF GPS present** — exact, subject to the tag being genuine.
- **Legible local text** (ccTLD, phone prefix, place name) — often
  country-level immediately, and town-level after geocoding.
- **Solar with a full timestamp** — tens of kilometres.
- **Scene model alone** — country-level at best, frequently wrong, and
  confidently so. Never rely on it unsupported.

Country labelling of grid cells uses nearest-populated-place rather than
polygon containment, which keeps the tool dependency-free and air-gappable.
Labels within ~50 km of a land border are unreliable. This affects the
human-readable label only, not the posterior.

---

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The astronomy and fusion tests assert against published ground truth
(J2000.0, solstice sun angles, known great-circle distances) rather than
values captured from this implementation, so they catch regressions instead
of freezing current behaviour.

---

## Layout

```
src/geoloc/
├── config.py         network posture, kill-switch, device selection
├── models.py         Evidence / GeoConstraint / Candidate / CaseReport
├── pipeline.py       orchestration
├── analyzers/        metadata · text_ocr · clip_scene · solar · faces
├── geo/              grid · fusion · overpass
├── viz.py            posterior rendering
├── report.py         self-contained HTML report
├── cli.py            command line
└── server.py + web/  local workbench
```

Adding an analyzer means emitting `Evidence` carrying `GeoConstraint`s;
fusion, ranking, reporting and the UI pick it up with no further changes.

---

## Licence

MIT — see [LICENSE](LICENSE).

Leaflet is vendored under BSD-2-Clause so the workbench runs with no CDN.
Model weights, OSM services and GeoNames data carry their own terms; all of
it is set out in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
