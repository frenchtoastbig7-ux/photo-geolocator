'use strict';

const $ = id => document.getElementById(id);
const state = {
  file: null, case: null, cells: [], settings: null,
  layers: { cells: null, markers: null, osm: null },
  selected: null, features: new Set(), tileLayer: null,
};

/* ── map ──────────────────────────────────────────────────────── */
const map = L.map('map', { worldCopyJump: true, zoomControl: true })
  .setView([25, 10], 2);
// The map div is sized by a CSS grid that settles after Leaflet initialises,
// so the initial tile fetch covers the wrong viewport. Re-measure once the
// layout lands, and on every subsequent resize.
requestAnimationFrame(() => map.invalidateSize());
window.addEventListener('load', () => map.invalidateSize());

// A map in a hidden or backgrounded tab measures zero, and Leaflet's
// getBoundsZoom then returns 0 -- so an analysis finishing while the analyst
// is looking at another tab would leave the map stuck at world view. Any fit
// requested against a zero-sized map is held and replayed once it has one.
let pendingFit = null;
function fitWhenSized(bounds, opts) {
  const size = map.getSize();
  if (size.x > 0 && size.y > 0) { pendingFit = null; map.fitBounds(bounds, opts); }
  else { pendingFit = { bounds, opts }; }
}
function flushPendingFit() {
  if (!pendingFit) return;
  const size = map.getSize();
  if (size.x > 0 && size.y > 0) {
    const { bounds, opts } = pendingFit;
    pendingFit = null;
    map.fitBounds(bounds, opts);
  }
}
new ResizeObserver(() => { map.invalidateSize(); flushPendingFit(); })
  .observe(document.getElementById('map'));
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) { map.invalidateSize(); flushPendingFit(); }
});

// Local-only tool: expose the map and analysis state so an analyst (or a
// test) can inspect what is actually plotted from the browser console.
window.geoloc = { map, get state() { return state; } };
state.layers.cells = L.layerGroup().addTo(map);
state.layers.markers = L.layerGroup().addTo(map);
state.layers.osm = L.layerGroup().addTo(map);

function setTiles(offline) {
  if (state.tileLayer) { map.removeLayer(state.tileLayer); state.tileLayer = null; }
  if (offline) return;                       // no tiles: no requests to a tile host
  state.tileLayer = L.tileLayer(
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }
  ).addTo(map);
}

/* Colour ramp mirrors the one used for the report PNG. */
const RAMP = [[26,60,120],[24,120,156],[54,178,140],[150,210,90],
              [238,214,70],[244,148,44],[222,68,40]];
function rampColor(t) {
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.floor(t), j = Math.min(i + 1, RAMP.length - 1), f = t - i;
  const c = RAMP[i].map((v, k) => Math.round(v * (1 - f) + RAMP[j][k] * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function drawPosterior(cells) {
  state.layers.cells.clearLayers();
  if (!cells || !cells.length) return;
  // Log scale: posteriors are extremely peaked, and a linear ramp renders
  // everything but the single best cell as invisible.
  const logs = cells.map(c => Math.log10(Math.max(c.rel, 1e-6)));
  const lo = Math.min(...logs), hi = Math.max(...logs), span = (hi - lo) || 1;
  cells.forEach((c, i) => {
    const t = (logs[i] - lo) / span;
    L.rectangle(c.bounds, {
      stroke: false, fillColor: rampColor(t),
      fillOpacity: 0.15 + 0.55 * t, interactive: false,
    }).addTo(state.layers.cells);
  });
}

function drawCandidates(cands) {
  state.layers.markers.clearLayers();
  cands.forEach(c => {
    const m = L.circleMarker([c.lat, c.lon], {
      radius: Math.max(6, 13 - c.rank), color: '#fff', weight: 2,
      fillColor: c.rank === 1 ? '#de4428' : '#1f6feb', fillOpacity: .9,
    }).bindPopup(
      `<strong>#${c.rank} ${esc(c.label)}</strong><br>` +
      `<code>${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}</code><br>` +
      `posterior ${(c.score * 100).toFixed(3)}%`
    );
    m.addTo(state.layers.markers);
    m.on('click', () => selectCandidate(c.rank));
  });
  if (cands.length) {
    const bounds = L.featureGroup(state.layers.markers.getLayers()).getBounds();
    if (bounds.isValid()) fitWhenSized(bounds.pad(0.35), { maxZoom: 9 });
  }
}

function selectCandidate(rank) {
  state.selected = rank;
  document.querySelectorAll('.cand').forEach(el =>
    el.classList.toggle('sel', +el.dataset.rank === rank));
  const c = (state.case?.candidates || []).find(x => x.rank === rank);
  if (c) map.setView([c.lat, c.lon], Math.max(map.getZoom(), 8));
}

/* ── helpers ──────────────────────────────────────────────────── */
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));

function setStatus(msg, isErr) {
  const el = $('status');
  el.className = 'status' + (isErr ? ' err' : '');
  el.innerHTML = msg || '';
}

/* ── settings ─────────────────────────────────────────────────── */
async function loadSettings() {
  state.settings = await (await fetch('/api/settings')).json();
  $('offlineToggle').checked = state.settings.offline;
  $('devInfo').textContent =
    `${state.settings.device} · grid ${state.settings.grid_step}°`;
  applyOffline(state.settings.offline);
  renderFeaturePicker(state.settings.features || []);
}

function applyOffline(off) {
  setTiles(off);
  document.querySelector('.dot').classList.toggle('off', off);
  const b = $('netBanner');
  if (off) {
    b.textContent = 'Offline mode: no outbound requests. Map tiles, geocoding '
      + 'and OSM feature search are disabled. Analysis runs entirely locally.';
    b.classList.remove('hidden');
  } else {
    b.classList.add('hidden');
  }
}

$('offlineToggle').addEventListener('change', async e => {
  const r = await fetch('/api/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ offline: e.target.checked }),
  });
  state.settings = await r.json();
  applyOffline(state.settings.offline);
  refreshModel();
});

/* ── file input ───────────────────────────────────────────────── */
const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) acceptFile(f);
});
$('file').addEventListener('change', e => {
  if (e.target.files[0]) acceptFile(e.target.files[0]);
});

function acceptFile(f) {
  if (!f.type.startsWith('image/') && !/\.(jpe?g|png|tiff?|webp|heic|bmp)$/i.test(f.name)) {
    setStatus('Not an image file.', true); return;
  }
  state.file = f;
  const url = URL.createObjectURL(f);
  $('preview').src = url;
  $('preview').classList.remove('hidden');
  $('dropInner').classList.add('hidden');
  $('fileMeta').textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
  $('runBtn').disabled = false;
  setStatus('');
}

/* ── driving side segmented control ───────────────────────────── */
let drivingSide = '';
document.querySelectorAll('#drivingSide button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#drivingSide button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    drivingSide = b.dataset.v;
  });
});

/* live solar elevation readout from the height:shadow ratio */
function updateElev() {
  const h = parseFloat($('objH').value), s = parseFloat($('shadowLen').value);
  $('elevCalc').textContent = (h > 0 && s > 0)
    ? `Implied solar elevation: ${(Math.atan2(h, s) * 180 / Math.PI).toFixed(1)}°`
    : '';
}
$('objH').addEventListener('input', updateElev);
$('shadowLen').addEventListener('input', updateElev);

/* ── run analysis ─────────────────────────────────────────────── */
$('runBtn').addEventListener('click', async () => {
  if (!state.file) return;
  const fd = new FormData();
  fd.append('image', state.file);
  fd.append('driving_side', drivingSide);
  fd.append('shadow_azimuth', $('shadowAz').value);
  fd.append('object_height', $('objH').value);
  fd.append('shadow_length', $('shadowLen').value);
  fd.append('capture_utc', $('captureUtc').value);
  fd.append('capture_date', $('captureDate').value);
  fd.append('countries', $('countries').value);
  fd.append('exclude', $('exclude').value);
  fd.append('notes', $('notes').value);
  fd.append('run_scene', $('runScene').checked ? 'true' : 'false');
  fd.append('habitation_prior', $('habPrior').checked ? 'true' : 'false');

  $('runBtn').disabled = true;
  setStatus('<span class="spinner"></span>Analysing — the scene model takes a few seconds…');
  try {
    const r = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    render(data);
    setStatus('Analysis complete.');
  } catch (err) {
    setStatus('Failed: ' + err.message, true);
  } finally {
    $('runBtn').disabled = false;
  }
});

/* ── render results ───────────────────────────────────────────── */
function render(data) {
  state.case = data.case;
  state.cells = data.cells;
  $('empty').classList.add('hidden');
  $('results').classList.remove('hidden');

  const sm = state.case.summary || {};
  $('brief').innerHTML = !sm.assessment ? '' : `
    <h2>Intelligence brief</h2>
    <div class="brief">
      <p class="assess">${esc(sm.assessment)}</p>
      <table class="kv">
        <tr><th>Location</th><td>${esc(sm.location || '—')}</td></tr>
        <tr><th>Coordinates</th><td><code>${esc(sm.coordinates || '—')}</code></td></tr>
        <tr><th>Countries</th><td>${esc((sm.countries || []).join(', ') || '—')}</td></tr>
        <tr><th>Time of day</th><td>${esc((sm.time_of_day || {}).text || '—')}</td></tr>
        <tr><th>People</th><td>${sm.people_present || 0} present (${sm.faces_visible || 0} with a visible face)</td></tr>
      </table>
      ${sm.description ? `<p class="desc">${esc(sm.description)}</p>` : ''}
      ${(sm.provenance || []).length ? '<p class="lbl">Provenance</p><ul>' +
        sm.provenance.map(n => `<li>${esc(n)}</li>`).join('') + '</ul>' : ''}
      ${(sm.next_steps || []).length ? '<p class="lbl">Recommended next steps</p><ol>' +
        sm.next_steps.map(n => `<li>${esc(n)}</li>`).join('') + '</ol>' : ''}
    </div>`;

  $('warnings').innerHTML = (state.case.warnings || [])
    .map(w => `<div class="warn-box">${esc(w)}</div>`).join('');

  const e = state.case.entropy_bits;
  const area = state.case.credible_area_km2;
  const band = state.case.precision_band || '';
  const weak = band === 'country' || band === 'unconstrained';
  const panels = renderMetaVerdict(state.case) + renderCaptureTime(state.case)
                 + renderMetaFields(state.case);
  const host = $('operatorPanels');
  if (host) host.innerHTML = panels;
  $('entropy').innerHTML = area == null ? '' : `
    <div class="precision ${weak ? 'weak' : 'ok'}">
      <strong>Precision: ${esc(band.toUpperCase())}</strong>
      — 90% of the posterior covers ${Math.round(area).toLocaleString()} km²
      <div class="pnote">${esc(state.case.precision_note || '')}</div>
    </div>
    <div class="muted small" style="margin-top:.3rem">
      Posterior entropy ${e == null ? '—' : e.toFixed(2)} bits.
    </div>`;

  const candHeading = weak
    ? '<p class="warn-box">The candidates below are the most populated points '
      + 'inside a very large area. They are NOT evidence-backed locations.</p>'
    : '';
  $('candidates').innerHTML = candHeading + (state.case.candidates || []).map(c => `
    <div class="cand" data-rank="${c.rank}">
      <span class="rank">${c.rank}</span>
      <span class="body">
        <div class="lbl">${esc(c.label || 'unnamed')}</div>
        <div class="co">${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}</div>
      </span>
      <span class="pct">${(c.score * 100).toFixed(2)}%</span>
    </div>`).join('') || '<p class="muted small">No candidates above threshold.</p>';
  document.querySelectorAll('.cand').forEach(el =>
    el.addEventListener('click', () => selectCandidate(+el.dataset.rank)));

  const tc = state.case.top_countries || [];
  const max = tc.length ? tc[0][1] : 1;
  $('countryBars').innerHTML = tc.map(([cc, v]) => `
    <div class="bar"><span class="cc">${esc(cc)}</span>
      <span class="track"><span class="fill" style="width:${(v / max * 100).toFixed(1)}%"></span></span>
      <span class="v">${(v * 100).toFixed(1)}%</span></div>`).join('');

  $('evidence').innerHTML = (state.case.evidence || []).map(ev => `
    <div class="ev">
      <div class="ev-hd">
        <span class="ev-t">${esc(ev.title)}</span>
        <span class="pill ${ev.confidence}">${ev.confidence}</span>
        <span class="pill">${esc(ev.analyzer)}</span>
      </div>
      ${ev.detail ? `<div class="ev-d">${esc(ev.detail)}</div>` : ''}
    </div>`).join('');

  const link = $('reportLink');
  link.href = `/api/case/${encodeURIComponent(state.case.case_id)}/file/report.html`;
  link.textContent = 'Open full report';
  link.classList.remove('hidden');

  drawPosterior(state.cells);
  drawCandidates(state.case.candidates || []);
  state.layers.osm.clearLayers();
  $('osmResults').innerHTML = '';
}

/* ── OSM refinement ───────────────────────────────────────────── */
function renderFeaturePicker(features) {
  $('featurePicker').innerHTML = '<div class="chips">' + features.map(f =>
    `<span class="chip" data-f="${esc(f)}">${esc(f)}</span>`).join('') + '</div>';
  document.querySelectorAll('.chip').forEach(ch => ch.addEventListener('click', () => {
    const f = ch.dataset.f;
    if (state.features.has(f)) { state.features.delete(f); ch.classList.remove('on'); }
    else { state.features.add(f); ch.classList.add('on'); }
  }));
}

$('geoBtn').addEventListener('click', async () => {
  const q = $('geoQ').value.trim();
  if (!q) return;
  const countries = (state.case?.top_countries || []).slice(0, 5).map(x => x[0]);
  $('osmResults').innerHTML = '<p class="muted small"><span class="spinner"></span>Geocoding…</p>';
  try {
    const r = await fetch('/api/osm/geocode', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q, countries }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    showOsm(d.results.map(x => ({
      lat: x.lat, lon: x.lon, name: x.display_name,
      sub: `${x.category}/${x.type} · ${x.country_code}`, url: x.url,
    })));
  } catch (err) {
    $('osmResults').innerHTML = `<p class="warn-box">${esc(err.message)}</p>`;
  }
});

$('featBtn').addEventListener('click', async () => {
  const c = (state.case?.candidates || []).find(x => x.rank === (state.selected || 1));
  if (!c) { $('osmResults').innerHTML = '<p class="muted small">Analyse an image first.</p>'; return; }
  if (!state.features.size) { $('osmResults').innerHTML = '<p class="muted small">Select at least one feature above.</p>'; return; }
  $('osmResults').innerHTML = '<p class="muted small"><span class="spinner"></span>Querying Overpass…</p>';
  try {
    const r = await fetch('/api/osm/features', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: c.lat, lon: c.lon, radius_km: 25,
                             features: [...state.features] }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    showOsm(d.results.map(x => ({
      lat: x.lat, lon: x.lon, name: x.name || '(unnamed)',
      sub: Object.entries(x.tags).slice(0, 3).map(([k, v]) => `${k}=${v}`).join(' · '),
      url: x.url,
    })), `${d.count} feature(s) within 25 km of #${c.rank}`);
  } catch (err) {
    $('osmResults').innerHTML = `<p class="warn-box">${esc(err.message)}</p>`;
  }
});

function showOsm(items, heading) {
  state.layers.osm.clearLayers();
  if (!items.length) { $('osmResults').innerHTML = '<p class="muted small">No matches.</p>'; return; }
  items.slice(0, 60).forEach(it => {
    L.circleMarker([it.lat, it.lon], {
      radius: 4, color: '#0b8a5b', weight: 2, fillOpacity: .8, fillColor: '#3ecf8e',
    }).bindPopup(`<strong>${esc(it.name)}</strong><br>${esc(it.sub)}`)
      .addTo(state.layers.osm);
  });
  $('osmResults').innerHTML = (heading ? `<p class="muted small">${esc(heading)}</p>` : '')
    + items.slice(0, 40).map(it => `
      <div class="osm-item" data-lat="${it.lat}" data-lon="${it.lon}">
        <strong>${esc(it.name)}</strong><br>
        <span class="muted">${esc(it.sub)}</span>
        ${it.url ? ` · <a href="${esc(it.url)}" target="_blank" rel="noreferrer">OSM</a>` : ''}
      </div>`).join('');
  document.querySelectorAll('.osm-item').forEach(el => el.addEventListener('click', e => {
    if (e.target.tagName === 'A') return;
    map.setView([+el.dataset.lat, +el.dataset.lon], 15);
  }));
}

/* ── optional scene model ─────────────────────────────────────── */
const fmtGB = b => (b / 1e9).toFixed(2) + ' GB';
let modelPoll = null;

async function refreshModel() {
  let m;
  try { m = await (await fetch('/api/model')).json(); }
  catch { return; }
  state.model = m;
  renderModelBox(m);

  // Poll only while a download is actually in flight.
  const running = m.progress && m.progress.state === 'running';
  if (running && !modelPoll) modelPoll = setInterval(refreshModel, 1000);
  if (!running && modelPoll) { clearInterval(modelPoll); modelPoll = null; }
}

function renderModelBox(m) {
  const box = $('modelBox');
  const runScene = $('runScene');
  box.classList.remove('hidden');

  if (!m.torch_available) {
    box.className = 'modelbox warn';
    box.innerHTML = `<div class="mb-hd"><span class="mb-dot missing"></span>
      <span class="mb-title">Scene model unavailable</span></div>
      <p>PyTorch is not present in this build, so StreetCLIP cannot run.
      Every other analyzer works normally.</p>`;
    runScene.checked = false; runScene.disabled = true;
    return;
  }

  const p = m.progress || {};
  if (p.state === 'running') {
    box.className = 'modelbox';
    box.innerHTML = `<div class="mb-hd"><span class="mb-dot"></span>
      <span class="mb-title">Installing scene model…</span></div>
      <div class="progress"><span class="fill" style="width:${p.percent}%"></span></div>
      <p>${fmtGB(p.downloaded)} of ${fmtGB(p.total)} · ${p.percent}%
      — you can run analyses meanwhile; the scene model joins in once ready.</p>`;
    runScene.disabled = true;
    return;
  }

  if (m.installed) {
    box.className = 'modelbox';
    box.innerHTML = `<div class="mb-hd"><span class="mb-dot ok"></span>
      <span class="mb-title">Scene model installed</span></div>
      <p>StreetCLIP ${esc(m.revision)} · runs entirely offline.</p>
      <div class="mb-actions"><button class="linklike" id="modelRemove">Remove and free ${fmtGB(m.approx_bytes)}</button></div>`;
    runScene.disabled = false;
    $('modelRemove').addEventListener('click', async () => {
      if (!confirm('Delete the downloaded scene model weights?')) return;
      await fetch('/api/model', { method: 'DELETE' });
      refreshModel();
    });
    return;
  }

  // Not installed.
  box.className = 'modelbox warn';
  const err = p.state === 'error'
    ? `<p><strong>Last attempt failed:</strong> ${esc(p.error)}</p>` : '';
  box.innerHTML = `<div class="mb-hd"><span class="mb-dot missing"></span>
    <span class="mb-title">Scene model not installed</span></div>
    <p>A one-time ${fmtGB(m.approx_bytes)} download. Metadata, OCR, solar
    geometry and fusion all work without it — StreetCLIP adds a coarse
    country estimate and scene description.</p>${err}
    <div class="mb-actions"><button class="ghost" id="modelInstall">Install scene model</button></div>`;
  runScene.checked = false; runScene.disabled = true;
  $('modelInstall').addEventListener('click', async () => {
    const btn = $('modelInstall');
    btn.disabled = true; btn.textContent = 'Starting…';
    try {
      const r = await fetch('/api/model/install', { method: 'POST' });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      refreshModel();
    } catch (e) {
      btn.disabled = false; btn.textContent = 'Install scene model';
      box.insertAdjacentHTML('beforeend',
        `<p><strong>${esc(e.message)}</strong></p>`);
    }
  });
}

/* ── operator panels: metadata, spoof check, capture time ─────── */
const VERDICT_STYLE = {
  consistent:   ['ok',   'Metadata consistent'],
  conflict:     ['bad',  'METADATA CONFLICT'],
  unverifiable: ['warn', 'Metadata unverifiable'],
  no_metadata:  ['warn', 'No GPS metadata'],
};

function renderMetaVerdict(c) {
  const v = c.metadata_verdict || {};
  if (!v.verdict) return '';
  const [cls, label] = VERDICT_STYLE[v.verdict] || ['warn', v.verdict];
  const rows = [];
  if (v.gps) rows.push(['Tagged position', `${v.gps[0].toFixed(5)}, ${v.gps[1].toFixed(5)}`]);
  if (v.content_best) rows.push(['Image content suggests', `${v.content_best[0].toFixed(3)}, ${v.content_best[1].toFixed(3)}`]);
  if (v.distance_km != null) rows.push(['Separation', `${v.distance_km.toLocaleString()} km`]);
  if (v.gps_percentile != null) rows.push(['Content support for the tag', `${v.gps_percentile}th percentile`]);
  return `
    <section class="panel">
      <h3>Metadata verification</h3>
      <div class="verdict ${cls}"><strong>${esc(label)}</strong>
        <div class="vsub">${esc(v.headline || '')}</div></div>
      <p class="muted small">${esc(v.detail || '')}</p>
      ${rows.length ? `<table class="kv">${rows.map(r =>
        `<tr><th>${esc(r[0])}</th><td>${esc(String(r[1]))}</td></tr>`).join('')}</table>` : ''}
      ${(v.indicators || []).length ? `<ul class="ind">${
        v.indicators.map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : ''}
    </section>`;
}

function renderMetaFields(c) {
  const f = c.metadata_fields || {};
  const keys = Object.keys(f);
  if (!keys.length) return `
    <section class="panel"><h3>File metadata</h3>
      <p class="muted small">No metadata fields present — the file has been
      stripped or re-encoded.</p></section>`;
  return `
    <section class="panel">
      <h3>File metadata <span class="muted small">(${keys.length} fields)</span></h3>
      <details><summary>Show all fields</summary>
        <table class="kv meta">${keys.sort().map(k =>
          `<tr><th>${esc(k)}</th><td>${esc(String(f[k]))}</td></tr>`).join('')}</table>
      </details>
    </section>`;
}

function renderCaptureTime(c) {
  const sm = c.summary || {};
  const t = sm.time_of_day || {};
  const st = c.solar_timing || {};
  const m = st.measurement || {};
  const conf = m.confidence ? ` <span class="pill ${m.confidence}">${m.confidence}</span>` : '';
  return `
    <section class="panel">
      <h3>Capture time${conf}</h3>
      <p class="small">${esc(t.text || 'Not determinable from this image.')}</p>
      ${t.shadow_check ? `<p class="muted small"><strong>Shadow cross-check:</strong> ${esc(t.shadow_check)}</p>` : ''}
      ${m.elevation_deg != null ? `<table class="kv">
        <tr><th>Sun elevation</th><td>${m.elevation_deg}&deg; above horizon</td></tr>
        <tr><th>Shadow : subject ratio</th><td>${m.ratio}</td></tr>
        <tr><th>Measured from</th><td>${esc(m.source || '')}</td></tr>
        ${st.date_window ? `<tr><th>Season implied</th><td>${esc(st.date_window[0])} &ndash; ${esc(st.date_window[1])}</td></tr>` : ''}
      </table>` : ''}
      ${(m.notes || []).length ? `<ul class="ind">${m.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : ''}
    </section>`;
}

/* ── case history ─────────────────────────────────────────────── */
$('historyBtn').addEventListener('click', async () => {
  $('modal').classList.remove('hidden');
  const cases = await (await fetch('/api/cases')).json();
  $('caseList').innerHTML = cases.length ? cases.map(c => `
    <div class="case-row" data-id="${esc(c.case_id)}">
      <span><strong>${esc(c.filename || c.case_id)}</strong><br>
        <span class="muted small">${esc(c.top || 'no candidate')}</span></span>
      <span class="muted small">${esc((c.created_utc || '').slice(0, 16).replace('T', ' '))}</span>
    </div>`).join('') : '<p class="muted small">No saved cases yet.</p>';
  document.querySelectorAll('.case-row').forEach(el => el.addEventListener('click', async () => {
    const d = await (await fetch(`/api/case/${encodeURIComponent(el.dataset.id)}`)).json();
    render(d);
    $('modal').classList.add('hidden');
  }));
});
$('modalClose').addEventListener('click', () => $('modal').classList.add('hidden'));
$('modal').addEventListener('click', e => {
  if (e.target.id === 'modal') $('modal').classList.add('hidden');
});

loadSettings();
refreshModel();
