"""`geoloc corpus ...` — build, inspect and calibrate the reference corpus."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Build and query the visual place-recognition corpus.")
console = Console()


def _sites_for(country: str | None, facility: str | None, near: str | None,
               radius_km: float, limit: int) -> list[dict]:
    """Resolve a site list from a country+facility, optionally near a point."""
    from ..geo import facility as facility_mod
    from ..geo.grid import haversine_km

    if not (country and facility):
        raise typer.BadParameter("--country and --facility are both required")
    fac = next((f for f in facility_mod.FACILITIES if f.key == facility), None)
    if fac is None:
        raise typer.BadParameter(f"unknown facility {facility!r}")

    raw = facility_mod.fetch_sites(country.upper(), fac)
    sites = [{"id": f"{country.upper()}_{facility}_{i}",
              "name": s.get("name") or f"site {i}",
              "lat": s["lat"], "lon": s["lon"]}
             for i, s in enumerate(raw)]

    if near:
        try:
            lat_s, lon_s = near.split(",")
            lat, lon = float(lat_s), float(lon_s)
        except ValueError as exc:
            raise typer.BadParameter("--near must be 'lat,lon'") from exc
        sites = [s for s in sites
                 if haversine_km(lat, lon, s["lat"], s["lon"]) <= radius_km]
        sites.sort(key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))

    named = [s for s in sites if s["name"] and not s["name"].startswith("site ")]
    return (named or sites)[:limit]


@app.command("build")
def build_corpus(
    area: str = typer.Argument(..., help="Name for this corpus, e.g. 'fr-rail-south'."),
    country: str = typer.Option(None, "--country", help="ISO2 country code."),
    facility: str = typer.Option(None, "--facility", help="Facility type (see list-facilities)."),
    near: str = typer.Option(None, "--near", help="'lat,lon' to restrict the area."),
    radius_km: float = typer.Option(60.0, "--radius-km"),
    limit: int = typer.Option(40, "--limit", help="Maximum sites to harvest."),
    per_site: int = typer.Option(60, "--per-site", help="Reference images per site."),
    site_radius_m: int = typer.Option(500, "--site-radius-m"),
    index_only: bool = typer.Option(False, "--index-only", help="Skip harvesting; re-embed what is on disk."),
):
    """Harvest reference imagery for an area and build its descriptor index.

    Coverage density is what determines whether matching works at all: with a
    single image per site a wrong answer wins confidently, so --per-site is
    the setting that matters most.
    """
    from . import corpus
    from . import index as index_mod

    if not index_only:
        sites = _sites_for(country, facility, near, radius_km, limit)
        if not sites:
            console.print("[red]No sites resolved for those filters.[/]")
            raise typer.Exit(1)
        console.print(f"Harvesting up to {per_site} image(s) for each of "
                      f"{len(sites)} site(s) into area [cyan]{area}[/].")
        console.print("[dim]Commons is rate-limited, so this is deliberately "
                      "slow and is resumable — interrupt it safely.[/]")
        with console.status("[cyan]harvesting[/]") as status:
            records, stats = corpus.harvest(
                area, sites, per_site=per_site, radius_m=site_radius_m,
                progress=lambda m: status.update(f"[cyan]{m}[/]"))
        console.print(f"[green]Harvest:[/] {stats.downloaded} downloaded, "
                      f"{stats.skipped} already present, {stats.failed} failed, "
                      f"{len(records)} images total.")

    with console.status("[cyan]embedding[/]") as status:
        idx = index_mod.build(area, progress=lambda m: status.update(f"[cyan]{m}[/]"))
    console.print(f"[green]Index:[/] {len(idx)} descriptors across "
                  f"{len(idx.sites)} site(s) → {index_mod.index_path(area)}")


@app.command("list")
def list_corpora():
    """Show harvested areas."""
    from . import corpus
    from . import index as index_mod

    t = Table(title="VPR corpora")
    t.add_column("area")
    t.add_column("images", justify="right")
    t.add_column("sites", justify="right")
    t.add_column("indexed", justify="right")
    any_row = False
    for name, count in corpus.list_areas():
        idx = index_mod.load(name)
        sites = len({r["site_id"] for r in corpus.load_manifest(name)})
        t.add_row(name, str(count), str(sites), str(len(idx)) if idx else "—")
        any_row = True
    console.print(t if any_row else "[dim]No corpora yet.[/]")


@app.command("match")
def match_image(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    area: str = typer.Option(..., "--area"),
    top_k: int = typer.Option(25, "--top-k"),
):
    """Match one image against a corpus, showing the abstention decision."""
    from . import index as index_mod
    from . import match as match_mod

    idx = index_mod.load(area)
    if idx is None:
        console.print(f"[red]No index for area {area!r}.[/] Build one first.")
        raise typer.Exit(1)
    result = match_mod.query_image(idx, image, top_k=top_k)

    style = "green" if result.accepted else "yellow"
    console.print(f"[{style}]{'MATCH' if result.accepted else 'NO MATCH'}[/] — {result.reason}\n")
    t = Table(title=f"Nearest sites in {area}")
    t.add_column("#", justify="right")
    t.add_column("site")
    t.add_column("score", justify="right")
    t.add_column("support", justify="right")
    t.add_column("coords")
    for i, m in enumerate(result.matches[:8], 1):
        t.add_row(str(i), m.site_name or m.site_id, f"{m.best_score:.4f}",
                  str(m.support), f"{m.lat:.4f}, {m.lon:.4f}")
    console.print(t)


@app.command("calibrate")
def calibrate(
    area: str = typer.Option(..., "--area"),
    per_site: int = typer.Option(6, "--per-site", help="Query images per held-out site."),
):
    """Measure how often this corpus fabricates an identification.

    Leave-one-SITE-out, not leave-one-image-out. The distinction is the whole
    point: holding out a single image leaves its site in the index and only
    measures ranking among known places, which is not the task. In practice
    the photographed place is usually absent from the corpus entirely, and
    what matters is whether the tool says so.

    Two figures are reported:

    * open-set false accepts -- queries whose site was removed wholesale.
      Every acceptance here is a fabricated identification, and this number
      is the one to judge a corpus by.
    * closed-set correct accepts -- queries whose site remains. This measures
      how much genuine signal the thresholds discard, i.e. the cost of
      caution.
    """
    import numpy as np

    from . import index as index_mod
    from . import match as match_mod

    idx = index_mod.load(area)
    if idx is None:
        console.print(f"[red]No index for area {area!r}.[/]")
        raise typer.Exit(1)

    class _Sub:
        def __init__(self, desc, recs):
            self.descriptors, self.records = desc, recs

        def __len__(self):
            return len(self.records)

        def search(self, q, top_k=25):
            sims = self.descriptors @ q
            k = min(top_k, sims.size)
            i = np.argpartition(sims, -k)[-k:]
            i = i[np.argsort(sims[i])[::-1]]
            return [(float(sims[j]), self.records[j]) for j in i]

    sites = sorted({r["site_id"] for r in idx.records})
    open_n = open_fa = closed_n = closed_ok = 0
    for site in sites:
        keep = np.array([r["site_id"] != site for r in idx.records])
        held = [i for i, r in enumerate(idx.records) if r["site_id"] == site]
        if len(held) < 2:
            continue
        absent = _Sub(idx.descriptors[keep],
                      [r for r, m in zip(idx.records, keep, strict=True) if m])
        for qi in held[:per_site]:
            q = idx.descriptors[qi]
            res = match_mod.match(absent, q)
            open_n += 1
            open_fa += bool(res.accepted)

            present = keep.copy()
            present[held] = True
            present[qi] = False
            sub = _Sub(idx.descriptors[present],
                       [r for r, m in zip(idx.records, present, strict=True) if m])
            r2 = match_mod.match(sub, q)
            closed_n += 1
            closed_ok += bool(r2.accepted and r2.matches
                              and r2.matches[0].site_id == site)

    if not open_n:
        console.print("[yellow]Every site has fewer than two images; harvest "
                      "more per site before calibrating.[/]")
        raise typer.Exit(1)

    fa_rate = open_fa / open_n
    t = Table(title=f"Calibration for '{area}' ({len(idx)} images, {len(sites)} sites)")
    t.add_column("test")
    t.add_column("queries", justify="right")
    t.add_column("result", justify="right")
    t.add_row("open set — site absent, must reject", str(open_n),
              f"{open_fa} fabricated ({fa_rate * 100:.0f}%)")
    t.add_row("closed set — site present, should find it", str(closed_n),
              f"{closed_ok} correct ({closed_ok / max(closed_n,1) * 100:.0f}%)")
    console.print(t)

    if fa_rate > 0.05:
        console.print(f"\n[red]This corpus fabricates on {fa_rate*100:.0f}% of "
                      "absent-site queries.[/] Raise MIN_SIMILARITY or harvest "
                      "denser coverage before trusting its matches.")
    else:
        console.print(f"\n[green]Fabrication rate {fa_rate*100:.1f}%[/] — "
                      "acceptable. Note the closed-set figure is the cost of "
                      "that caution: genuine matches are discarded to buy it.")
    console.print(f"[dim]Thresholds: similarity floor {match_mod.MIN_SIMILARITY}, "
                  f"margin {match_mod.MIN_MARGIN}, ratio {match_mod.MIN_RATIO}[/]")
