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
    """Resolve a site list for the CLI; the logic lives in vpr.sites."""
    from . import sites as sites_mod

    if not (country and facility):
        raise typer.BadParameter("--country and --facility are both required")
    point = None
    if near:
        try:
            lat_s, lon_s = near.split(",")
            point = (float(lat_s), float(lon_s))
        except ValueError as exc:
            raise typer.BadParameter("--near must be 'lat,lon'") from exc
    try:
        return sites_mod.resolve(country, facility, point, radius_km, limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    use_categories: bool = typer.Option(False, "--categories",
                                        help="Also harvest Commons categories. Measured "
                                             "WORSE than the radius default (14% vs 45% "
                                             "recall); opt in only for areas whose "
                                             "category trees are genuinely photographic."),
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
                use_categories=use_categories,
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
    """Measure how often this corpus names a place it does not contain.

    Leave-one-site-out; see geoloc.vpr.calibration for why that matters. The
    result is saved beside the index, where the workbench shows it.
    """
    from . import calibration

    try:
        result = calibration.run(area, per_site=per_site)
    except FileNotFoundError:
        console.print(f"[red]No index for area {area!r}.[/]")
        raise typer.Exit(1) from None

    if result["verdict"] == "insufficient":
        console.print(f"[yellow]{result['message']}[/]")
        raise typer.Exit(1)

    t = Table(title=(f"Calibration for '{area}' ({result['images']} images, "
                     f"{result['sites']} sites)"))
    t.add_column("test")
    t.add_column("queries", justify="right")
    t.add_column("result", justify="right")
    t.add_row("open set — site absent, must reject", str(result["open_queries"]),
              f"{result['fabricated']} fabricated ({result['fabrication_rate']:.0%})")
    t.add_row("closed set — site present, should find it",
              str(result["closed_queries"]),
              f"{result['correct']} correct ({result['recall']:.0%})")
    console.print(t)

    style = "red" if result["verdict"] == "too_high" else "green"
    console.print(f"\n[{style}]{result['message']}[/]")
    th = result["thresholds"]
    console.print(f"[dim]Thresholds: similarity floor {th['min_similarity']}, "
                  f"margin {th['min_margin']}, ratio {th['min_ratio']}. "
                  "Saved for the workbench.[/]")
