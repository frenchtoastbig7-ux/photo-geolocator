"""Self-contained HTML case report.

Everything is inlined as base64 so the file can be attached to a case record
and opened years later with no server, no network and no dependency on this
tool still existing. Reports embed the SHA-256 of the source image so the
analysis can be tied back to a specific file.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

from .models import CaseReport

_CSS = """
:root{--bg:#fbfbfa;--fg:#16181d;--muted:#5d6470;--line:#e2e4e9;--card:#fff;
--accent:#1f6feb;--warn:#b45309;--warn-bg:#fffbeb;--crit:#b91c1c}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e8ec;--muted:#9aa2b1;
--line:#252a33;--card:#161a21;--accent:#589bff;--warn:#fbbf24;--warn-bg:#2a2010;--crit:#f87171}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem} h2{font-size:1.1rem;margin:2.5rem 0 .75rem;
padding-bottom:.4rem;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);font-size:.85rem;margin-bottom:2rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem 1.1rem;margin-bottom:.75rem}
.warn{background:var(--warn-bg);border-color:var(--warn);color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{text-align:left;padding:.55rem .6rem;border-bottom:1px solid var(--line);
vertical-align:top} th{color:var(--muted);font-weight:600;font-size:.78rem;
text-transform:uppercase;letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{display:inline-block;font-size:.7rem;padding:.12rem .45rem;border-radius:99px;
border:1px solid var(--line);color:var(--muted);margin-right:.25rem}
.c-certain{color:#059669;font-weight:600}.c-high{color:var(--accent);font-weight:600}
.c-medium{color:var(--muted)}.c-low{color:var(--muted);opacity:.8}
.c-speculative{color:var(--muted);opacity:.6;font-style:italic}
img.shot{max-width:100%;border-radius:8px;border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.detail{color:var(--muted);font-size:.86rem;white-space:pre-wrap;margin-top:.3rem}
ul,ol{margin:.3rem 0 .3rem 1.2rem}
code{background:rgba(127,127,127,.12);padding:.1rem .3rem;border-radius:4px;
font-size:.85em;word-break:break-all}
.scroll{overflow-x:auto}
footer{margin-top:3rem;color:var(--muted);font-size:.78rem;border-top:1px solid var(--line);padding-top:1rem}
"""


def _b64(path: Path) -> str | None:
    try:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    except OSError:
        return None


def _esc(s) -> str:
    return html.escape(str(s))


def render_report(report: CaseReport, case_dir: Path) -> str:
    parts: list[str] = []
    a = parts.append

    a(f"<!-- geoloc case {report.case_id} -->")
    a(f"<style>{_CSS}</style><div class='wrap'>")
    a(f"<h1>Geolocation case {_esc(report.case_id)}</h1>")
    a(f"<div class='sub'>Generated {_esc(report.created_utc.isoformat())} &middot; "
      f"{'OFFLINE' if report.offline_mode else 'network-enabled'} mode &middot; "
      f"analyzers: {_esc(', '.join(report.analyzers_run) or 'none')}</div>")

    # ---- warnings -------------------------------------------------------
    for w in report.warnings:
        a(f"<div class='card warn'><strong>Caution</strong><div class='detail'>{_esc(w)}</div></div>")

    # ---- intelligence brief ------------------------------------------------
    sm = report.summary or {}
    if sm:
        a("<h2>Intelligence brief</h2><div class='card'>")
        a(f"<p><strong>{_esc(sm.get('assessment',''))}</strong></p><table>")
        for k, v in (("Location", sm.get("location", "")),
                     ("Coordinates", sm.get("coordinates", "")),
                     ("Countries", ", ".join(sm.get("countries") or []) or "—"),
                     ("Time of day", (sm.get("time_of_day") or {}).get("text", "—")),
                     ("People present",
                      f"{sm.get('people_present', 0)} "
                      f"({sm.get('faces_visible', 0)} with a visible face)")):
            a(f"<tr><th>{k}</th><td>{_esc(v)}</td></tr>")
        a("</table>")
        if sm.get("description"):
            a(f"<div class='detail'>{_esc(sm['description'])}</div>")
        if sm.get("provenance"):
            a("<p><strong>Provenance</strong></p><ul>"
              + "".join(f"<li>{_esc(n)}</li>" for n in sm["provenance"]) + "</ul>")
        if sm.get("next_steps"):
            a("<p><strong>Recommended next steps</strong></p><ol>"
              + "".join(f"<li>{_esc(n)}</li>" for n in sm["next_steps"]) + "</ol>")
        a("</div>")

    # ---- imagery ---------------------------------------------------------
    a("<h2>Imagery</h2><div class='grid'>")
    shown = case_dir / (report.faces.blurred_export or report.image.filename)
    src = _b64(shown)
    if src:
        cap = ("faces redacted" if report.faces.blurred_export else "as supplied")
        a(f"<div><img class='shot' src='{src}' alt='source image'>"
          f"<div class='detail'>Source image ({cap})</div></div>")
    post = _b64(case_dir / (report.posterior_png or "posterior.png"))
    if post:
        a(f"<div><img class='shot' src='{post}' alt='posterior heatmap'>"
          "<div class='detail'>Fused posterior, equirectangular projection. "
          "Log-scaled: warm areas carry more probability mass.</div></div>")
    a("</div>")

    # ---- file facts ------------------------------------------------------
    im = report.image
    a("<h2>File</h2><div class='card scroll'><table>")
    for k, v in (("Filename", im.filename), ("SHA-256", f"<code>{_esc(im.sha256)}</code>"),
                 ("Size", f"{im.bytes:,} bytes"),
                 ("Dimensions", f"{im.width} x {im.height} ({im.format}, {im.mode})"),
                 ("Path", f"<code>{_esc(im.path)}</code>")):
        a(f"<tr><th>{k}</th><td>{v if k in ('SHA-256','Path') else _esc(v)}</td></tr>")
    a("</table></div>")

    # ---- candidates ------------------------------------------------------
    a("<h2>Ranked candidates</h2>")
    if report.credible_area_km2 is not None:
        weak = report.precision_band in {"country", "unconstrained"}
        cls = "card warn" if weak else "card"
        a(f"<div class='{cls}'><strong>Precision: "
          f"{_esc(report.precision_band.upper())}</strong> &mdash; 90% of the "
          f"posterior covers {report.credible_area_km2:,.0f} km&sup2;"
          f"<div class='detail'>{_esc(report.precision_note)}</div></div>")
    if report.entropy_bits is not None:
        a(f"<div class='detail' style='margin-bottom:.6rem'>Posterior entropy "
          f"{report.entropy_bits:.2f} bits &mdash; lower means better constrained.</div>")
    a("<div class='card scroll'><table><tr><th>#</th><th>Location</th>"
      "<th>Coordinates</th><th class='num'>Posterior</th><th>Nearest place</th></tr>")
    for c in report.candidates:
        a(f"<tr><td class='num'>{c.rank}</td><td>{_esc(c.label)}</td>"
          f"<td><code>{c.lat:.4f}, {c.lon:.4f}</code></td>"
          f"<td class='num'>{c.score*100:.3f}%</td>"
          f"<td>{_esc(c.nearest_place)}"
          + (f" ({c.distance_to_place_km:.0f} km)" if c.distance_to_place_km else "")
          + "</td></tr>")
    if not report.candidates:
        a("<tr><td colspan='5'>No candidates produced.</td></tr>")
    a("</table></div>")

    # ---- country roll-up --------------------------------------------------
    if report.top_countries:
        a("<h2>Posterior mass by country</h2><div class='card scroll'><table>"
          "<tr><th>Country</th><th class='num'>Share</th></tr>")
        for cc, share in report.top_countries:
            a(f"<tr><td>{_esc(cc)}</td><td class='num'>{share*100:.1f}%</td></tr>")
        a("</table></div>")

    # ---- evidence ---------------------------------------------------------
    a("<h2>Evidence</h2>")
    for ev in report.evidence:
        cls = f"c-{ev.confidence.value}"
        tags = "".join(f"<span class='tag'>{_esc(t)}</span>" for t in ev.tags)
        a(f"<div class='card'><div><strong>{_esc(ev.title)}</strong> "
          f"<span class='{cls}'>[{ev.confidence.value}]</span> "
          f"<span class='tag'>{_esc(ev.analyzer)}</span>{tags}</div>")
        if ev.detail:
            a(f"<div class='detail'>{_esc(ev.detail)}</div>")
        if ev.constraints:
            notes = "; ".join(f"{c.kind.value}: {c.note}" for c in ev.constraints)
            a(f"<div class='detail'><em>Constraints applied &mdash; {_esc(notes)}</em></div>")
        a("</div>")

    if report.analyzers_skipped:
        a("<h2>Analyzers skipped</h2><div class='card scroll'><table>")
        for name, why in report.analyzers_skipped.items():
            a(f"<tr><th>{_esc(name)}</th><td>{_esc(why)}</td></tr>")
        a("</table></div>")

    a("<footer>Produced by a local single-image geolocation workbench. "
      "Candidates are probabilistic hypotheses derived from scene content, "
      "not confirmed locations; each requires independent corroboration "
      "before it is relied upon. No facial recognition was performed."
      "</footer></div>")
    return "\n".join(parts)


def write_report(report: CaseReport, case_dir: Path, dest: Path | None = None) -> Path:
    dest = dest or case_dir / "report.html"
    dest.write_text(render_report(report, case_dir), encoding="utf-8")
    return dest
