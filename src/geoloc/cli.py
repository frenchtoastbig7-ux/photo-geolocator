"""Command-line interface."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import SETTINGS
from .models import Confidence
from .pipeline import AnalystInput, analyze_image, load_case

app = typer.Typer(add_completion=False, help=__doc__,
                  context_settings={"help_option_names": ["-h", "--help"]})
console = Console()

_CONF_STYLE = {
    Confidence.CERTAIN: "bold green", Confidence.HIGH: "bold cyan",
    Confidence.MEDIUM: "white", Confidence.LOW: "dim",
    Confidence.SPECULATIVE: "dim italic",
}


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise typer.BadParameter(f"unrecognised datetime: {s!r}")


@app.command()
def analyze(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    offline: bool = typer.Option(False, "--offline", help="Block all outbound network traffic."),
    no_scene: bool = typer.Option(False, "--no-scene", help="Skip the StreetCLIP scene model."),
    no_prior: bool = typer.Option(False, "--no-population-prior",
                                  help="Disable the habitation prior (use for wilderness imagery)."),
    driving_side: str = typer.Option(None, "--driving-side", help="'left' or 'right', if you can determine it."),
    shadow_azimuth: float = typer.Option(None, "--shadow-azimuth",
                                         help="Shadow bearing, degrees clockwise from TRUE north."),
    sun_elevation: float = typer.Option(None, "--sun-elevation", help="Solar elevation in degrees."),
    height: float = typer.Option(None, "--object-height", help="Vertical object height (any unit)."),
    shadow_len: float = typer.Option(None, "--shadow-length", help="Shadow length (same unit)."),
    when: str = typer.Option(None, "--utc", help="Capture instant in UTC, e.g. 2024-06-21T14:30:00."),
    date: str = typer.Option(None, "--date", help="Capture date only, e.g. 2024-06-21."),
    country: list[str] = typer.Option(None, "--country", help="Restrict to ISO2 country code(s). Repeatable."),
    exclude: list[str] = typer.Option(None, "--exclude", help="Rule out ISO2 country code(s). Repeatable."),
    notes: str = typer.Option("", "--notes", help="Free-text analyst notes for the report."),
    facility: str = typer.Option(None, "--facility",
                                 help="Facility type you can identify (e.g. university, "
                                      "airport, hospital). Narrows to every such site in "
                                      "the candidate countries. Use --list-facilities."),
    candidates: int = typer.Option(8, "--candidates", "-n"),
    report: bool = typer.Option(True, "--report/--no-report", help="Write an HTML report."),
    json_out: bool = typer.Option(False, "--json", help="Print the full case JSON to stdout."),
):
    """Analyse a single image and produce ranked location hypotheses."""
    if offline:
        SETTINGS.offline = True

    if driving_side and driving_side.lower() not in {"left", "right"}:
        raise typer.BadParameter("--driving-side must be 'left' or 'right'")

    analyst = AnalystInput(
        driving_side=driving_side.lower() if driving_side else None,
        shadow_azimuth=shadow_azimuth, sun_elevation=sun_elevation,
        object_height=height, shadow_length=shadow_len,
        capture_datetime_utc=_parse_dt(when), capture_date=_parse_dt(date),
        notes=notes, facility_type=facility, country_hints=list(country or []),
        exclude_countries=list(exclude or []),
    )

    with console.status("[cyan]analysing[/]") as status:
        rep = analyze_image(
            image, analyst=analyst, run_scene_model=not no_scene,
            use_habitation_prior=not no_prior, n_candidates=candidates,
            progress=lambda m: status.update(f"[cyan]{m}[/]"),
        )

    case_dir = SETTINGS.case_dir / rep.case_id
    console.print(Panel(
        f"[bold]{rep.case_id}[/]\n{rep.image.filename}  "
        f"{rep.image.width}x{rep.image.height}\n"
        f"sha256 {rep.image.sha256[:32]}...\n"
        f"mode: {'OFFLINE' if rep.offline_mode else 'network-enabled'}",
        title="case", expand=False))

    for w in rep.warnings:
        # The precision warning gets its own prominent block below.
        if w.startswith("Precision:"):
            continue
        console.print(f"[yellow]![/] {w}\n")

    sm = rep.summary or {}
    if sm:
        console.print(Panel(
            f"[bold]{sm.get('assessment','')}[/]\n\n"
            f"[cyan]Location[/]     {sm.get('location','—')}\n"
            f"[cyan]Coordinates[/]  {sm.get('coordinates','—')}\n"
            f"[cyan]Countries[/]    {', '.join(sm.get('countries') or []) or '—'}\n"
            f"[cyan]Time of day[/]  {sm.get('time_of_day',{}).get('text','—')}\n"
            f"[cyan]People[/]       {sm.get('people_present',0)} face(s) detected\n\n"
            f"[cyan]Scene[/]\n{sm.get('description','—')}\n"
            + ("\n[cyan]Text observed[/]\n" + "; ".join(sm.get('observed_text') or [])
               if sm.get('observed_text') else "")
            + (("\n\n[cyan]Provenance[/]\n" + "\n".join(f"· {n}" for n in sm['provenance']))
               if sm.get('provenance') else "")
            + (("\n\n[cyan]Recommended next steps[/]\n"
                + "\n".join(f"{i}. {n}" for i, n in enumerate(sm['next_steps'], 1)))
               if sm.get('next_steps') else ""),
            title="Intelligence brief", expand=False))

    band_style = {
        "pinpoint": "bold green", "locality": "green", "regional": "yellow",
        "country": "bold red", "unconstrained": "bold red",
    }.get(rep.precision_band, "white")
    if rep.credible_area_km2 is not None:
        console.print(
            f"[{band_style}]Precision: {rep.precision_band.upper()}[/] — "
            f"90% of the posterior covers {rep.credible_area_km2:,.0f} km²\n"
            f"[dim]{rep.precision_note}[/]\n")

    title = "Ranked candidates"
    if rep.precision_band in {"country", "unconstrained"}:
        title = "Ranked candidates (NOT evidence-backed — see precision above)"
    t = Table(title=title, show_lines=False)
    t.add_column("#", justify="right")
    t.add_column("Location")
    t.add_column("Lat", justify="right")
    t.add_column("Lon", justify="right")
    t.add_column("Posterior", justify="right")
    for c in rep.candidates:
        t.add_row(str(c.rank), c.label, f"{c.lat:.4f}", f"{c.lon:.4f}",
                  f"{c.score*100:.3f}%")
    console.print(t)

    if rep.top_countries:
        ct = Table(title="Posterior mass by country")
        ct.add_column("Country")
        ct.add_column("Share", justify="right")
        for cc, share in rep.top_countries[:8]:
            ct.add_row(cc, f"{share*100:.1f}%")
        console.print(ct)

    et = Table(title="Evidence")
    et.add_column("Analyzer")
    et.add_column("Finding")
    et.add_column("Confidence")
    for ev in rep.evidence:
        et.add_row(ev.analyzer, ev.title,
                   f"[{_CONF_STYLE[ev.confidence]}]{ev.confidence.value}[/]")
    console.print(et)

    if rep.entropy_bits is not None:
        console.print(f"\n[dim]Posterior entropy: {rep.entropy_bits:.2f} bits "
                      "(lower = better constrained)[/]")

    if report:
        from .report import write_report
        p = write_report(rep, case_dir)
        # soft_wrap keeps paths on one line so they stay copy-pasteable in a
        # narrow terminal or when piped.
        console.print(f"[green]Report:[/] {p}", soft_wrap=True)
    console.print(f"[green]Case:[/]   {case_dir}", soft_wrap=True)

    if json_out:
        console.print_json(rep.model_dump_json())


@app.command("list-facilities")
def list_facilities():
    """Facility types accepted by `analyze --facility`."""
    from .geo.facility import FACILITIES

    t = Table(title="Facility types")
    t.add_column("key")
    t.add_column("description")
    t.add_column("triggered by text like")
    for f in FACILITIES:
        t.add_row(f.key, f.label, ", ".join(f.keywords[:5]) + " …")
    console.print(t)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address. Keep on loopback."),
    port: int = typer.Option(8711),
    offline: bool = typer.Option(False, "--offline", help="Block all outbound traffic."),
):
    """Launch the local web workbench."""
    import uvicorn

    if offline:
        SETTINGS.offline = True
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(f"[yellow]![/] Binding to {host} exposes this tool beyond "
                      "this machine. This is a local-only investigative tool; "
                      "do not expose it on a shared network.")
    console.print(f"[green]Workbench:[/] http://{host}:{port}")
    uvicorn.run("geoloc.server:app", host=host, port=port, log_level="warning")


@app.command("prefetch-models")
def prefetch_models():
    """Download model weights once so later runs can be fully air-gapped."""
    if SETTINGS.offline:
        console.print("[red]Offline mode is on; cannot prefetch.[/]")
        raise typer.Exit(1)
    from . import modelmgr

    if modelmgr.is_installed():
        console.print("[green]Already installed.[/] "
                      f"Cache: {modelmgr.hf_cache_dir()}")
        return

    console.print(f"Fetching [cyan]{modelmgr.MODEL_ID}[/] "
                  f"(~{modelmgr.APPROX_TOTAL_BYTES / 1e9:.1f} GB) "
                  f"to {modelmgr.hf_cache_dir()} ...")
    started = modelmgr.start_download()
    if not started.get("ok"):
        console.print(f"[red]{started.get('reason')}[/]")
        raise typer.Exit(1)

    import time

    from rich.progress import BarColumn, DownloadColumn, Progress, TimeRemainingColumn
    with Progress(BarColumn(), DownloadColumn(), TimeRemainingColumn(),
                  console=console) as bar:
        task = bar.add_task("download", total=modelmgr.APPROX_TOTAL_BYTES)
        while True:
            st = modelmgr.status()["progress"]
            bar.update(task, completed=st["downloaded"], total=st["total"])
            if st["state"] in {"done", "error"}:
                break
            time.sleep(0.5)

    final = modelmgr.status()["progress"]
    if final["state"] == "error":
        console.print(f"[red]Download failed:[/] {final['error']}")
        raise typer.Exit(1)
    console.print("[green]Done.[/] The scene model now runs offline.")


@app.command("show")
def show(case: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Print a previously saved case."""
    rep, _post = load_case(case)
    console.print_json(rep.model_dump_json())


if __name__ == "__main__":
    app()
