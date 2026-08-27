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

### macOS app (recommended)

Download `GeolocationWorkbench.dmg` from
[Releases](https://github.com/frenchtoastbig7-ux/photo-geolocator/releases),
drag the app to Applications, and open it. It carries its own Python and
PyTorch — nothing else to install.

**First launch:** the app is not notarised, so macOS will refuse a plain
double-click. Right-click it in Applications, choose **Open**, then confirm.
Only needed once.

The window is the workbench itself, not a browser tab. Analyses run in-process
on a random loopback port; nothing listens on a predictable port and nothing
is reachable from outside the machine.

Cases, logs and any downloaded model live in
`~/Library/Application Support/Geolocation Workbench`. Deleting that folder
removes everything the app has stored.

#### The optional scene model

The app ships without StreetCLIP's 1.7 GB weights, because bundling them
would triple the download. Everything else — metadata forensics, OCR, solar
geometry, fusion, mapping, reports, OSM refinement — works the moment you
install it.

To add the scene model, open **Analysis options → Install scene model**. It
downloads once, in-app, with a progress bar, and thereafter runs entirely
offline. You can remove it again from the same panel to reclaim the space.

### From source

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

## The intelligence brief

Every case opens with a digest rather than a coordinate list:

- **Assessment** — a location only when the precision band supports one. At
  country level it says so instead of naming a place.
- **Coordinates**, marked *indicative only* when not evidence-backed.
- **Time of day** — the EXIF timestamp if one survived; otherwise, given a
  measured shadow bearing and a candidate location, the sun is solved
  backwards to the instant that casts it (recovers a known 09:30 to within
  the 5-minute search step). If neither, it says so.
- **Description** — text actually read from the image, kept separate from
  scene-model readings, which are attributed and scored rather than asserted.
  That separation is deliberate: on a Gold Coast campus this model reported
  "hot desert" vegetation at 73% and "sub saharan african housing" at 57%.
- **Provenance** — camera, editing software, metadata stripping.
- **Recommended next steps**, chosen from how the case actually landed.

## Reading the output

Every case reports a **precision band** — the area of the smallest region
holding 90% of the posterior — because a ranked list looks equally confident
whether it came from a GPS tag or from nothing at all.

| Band | 90% region | What it means |
|---|---|---|
| `pinpoint` | < 25 km² | A specific site. |
| `locality` | < 2,500 km² | A town or suburb. Candidates are real leads. |
| `regional` | < 100,000 km² | A region. Candidates are areas to search. |
| `country` | < 3,000,000 km² | Country-level only. **Candidates are just the most populated points in a large area — not leads.** |
| `unconstrained` | larger | The image does not establish where it was taken. The list is noise. |

At `country` or `unconstrained` the candidate list is explicitly labelled as
not evidence-backed, in all three interfaces. This matters more than it
sounds: a scene-model-only result reaches `country` on almost any photo, and
the ranked cities underneath it carry no information whatsoever.

## What will not work, and why

**Naming a specific building or campus.** Asking the scene model to choose
between landmark names does not work, and fails in the most dangerous
possible way. Measured on this build:

| Image | Top "landmark" match |
|---|---|
| A real photo of Bond University | Bond University — 55.8% |
| A photo of a Croatian bakery | Bond University — **90.3%** |
| A blank grey image | Bond University — **59.7%** |

CLIP has a prior over the *strings*, not recognition of the *places*. Given
any candidate list it returns a confident winner regardless of the image, and
the controls score higher than the true positive. Landmark-name matching is
therefore deliberately not implemented; it would manufacture identifications.

Genuine landmark recognition needs image-to-image matching against reference
imagery of the area, which is a different capability from anything here.

**Refining within a country using the scene model.** Also measured and also
rejected: on a Gold Coast campus photo the model ranked Wollongong 25%,
Gold Coast 7%, and put Western Australia above Queensland. That is noise, and
folding it into the posterior would only add confident error.

**Ranking within a shortlist.** The facility gazetteer narrows a campus
photo to ~850 Australian universities, and the right one is in that list —
but nothing here can tell one campus from another. That needs reference
imagery of the candidates, which is the difference between this tool and a
reverse image search: a corpus, not a cleverer model.

**The reliable route to a specific place is text.** Legible distinctive
signage, geocoded, is what takes a case from `country` to `locality`. Generic
words cannot do it — "LIBRARY" geocodes to a library, just not the right one —
so they are filtered out rather than turned into a false constraint.

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

## Building the app

```bash
bash packaging/build_app.sh          # -> dist/Geolocation Workbench.app
bash packaging/make_dmg.sh           # -> dist/GeolocationWorkbench.dmg
```

`build_app.sh` regenerates the icon, runs PyInstaller against
`packaging/geoloc-mac.spec`, ad-hoc signs the bundle and clears the
quarantine flag. Ad-hoc signing is not notarisation — recipients still get
the unidentified-developer prompt on first open.

Two size decisions are encoded in the spec and are easy to undo if you
disagree with them:

- **StreetCLIP weights are excluded** and fetched on demand.
- **Three of geonamescache's four city datasets are excluded.** `grid.py`
  pins `min_city_population=15000`, so only `cities15000.json` is ever read;
  carrying the others would add ~170 MB for nothing. Changing that threshold
  means changing the spec too.

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
