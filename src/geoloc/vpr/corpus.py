"""Reference-imagery corpus, harvested from Wikimedia Commons.

The corpus is the whole capability. A place-recognition model can only find
what it has seen, so the difference between a tool that identifies a station
and one that shrugs is almost entirely how densely the candidate area is
covered -- which is exactly why a reverse image search over billions of
photographs succeeds where a local index of six pictures does not.

Measured here: with ONE reference image per station, DINOv2 confidently
ranked the wrong station first on the strength of a single lucky facade shot;
given several views each, that same wrong answer fell to last. Sparse
coverage does not merely fail, it fails confidently, so this harvester
prioritises depth per site over breadth across sites.

Commons is used because it is geotagged, licence-clean, needs no API key and
covers exactly the notable places an OSINT analyst encounters. It also rate
limits aggressively, so requests are batched, paced, cached and resumable.
"""
from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import SETTINGS, OfflineError

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Commons asks for modest concurrency; metadata is batched 50 at a time and
# thumbnails are fetched serially with a pause. Harvesting is a background
# chore that happens once per area, so slow is acceptable and being blocked
# halfway through is not.
METADATA_BATCH = 50
DOWNLOAD_PAUSE = 1.1
API_PAUSE = 1.5
THUMB_WIDTH = 640
MAX_RETRIES = 3


@dataclass
class HarvestStats:
    sites: int = 0
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"sites": self.sites, "discovered": self.discovered,
                "downloaded": self.downloaded, "skipped": self.skipped,
                "failed": self.failed, "notes": self.notes[-8:]}


def corpus_root() -> Path:
    return SETTINGS.model_cache / "vpr_corpus"


def area_dir(area: str) -> Path:
    safe = "".join(c for c in area if c.isalnum() or c in "-_")[:64] or "area"
    return corpus_root() / safe


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=120,
        headers={"User-Agent": SETTINGS.user_agent + " (VPR corpus builder)"},
        follow_redirects=True,
    )


def _get_json(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any] | None:
    """GET the Commons API, tolerating the HTML error pages it returns when
    throttling rather than crashing a long harvest."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(COMMONS_API, params=params)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return resp.json()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def discover_files(client: httpx.Client, lat: float, lon: float,
                   radius_m: int = 500, limit: int = 200) -> list[dict[str, Any]]:
    """Geotagged Commons files near a point, nearest first."""
    data = _get_json(client, {
        "action": "query", "format": "json", "list": "geosearch",
        "gscoord": f"{lat}|{lon}", "gsradius": str(radius_m),
        "gslimit": str(min(limit, 500)), "gsnamespace": "6",
    })
    if not data:
        return []
    return [
        {"title": x["title"], "lat": x.get("lat", lat), "lon": x.get("lon", lon),
         "dist_m": x.get("dist")}
        for x in data.get("query", {}).get("geosearch", [])
    ]


def resolve_thumbs(client: httpx.Client, titles: list[str]) -> dict[str, str]:
    """Thumbnail URLs for file titles, batched.

    One request per file is what gets a harvest throttled; the API accepts
    fifty titles at a time and there is no reason not to use it.
    """
    urls: dict[str, str] = {}
    for start in range(0, len(titles), METADATA_BATCH):
        chunk = titles[start:start + METADATA_BATCH]
        data = _get_json(client, {
            "action": "query", "format": "json", "prop": "imageinfo",
            "iiprop": "url|mime", "iiurlwidth": str(THUMB_WIDTH),
            "titles": "|".join(chunk),
        })
        if data:
            for page in data.get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                mime = info.get("mime", "")
                if not mime.startswith("image/"):
                    continue
                url = info.get("thumburl") or info.get("url")
                if url:
                    urls[page["title"]] = url
        time.sleep(API_PAUSE)
    return urls


def _image_path(site_dir: Path, title: str) -> Path:
    import hashlib

    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
    return site_dir / f"{digest}.jpg"


def harvest_site(client: httpx.Client, area: str, site: dict[str, Any],
                 *, per_site: int = 60, radius_m: int = 500,
                 stats: HarvestStats | None = None) -> list[dict[str, Any]]:
    """Fetch reference imagery for one site. Resumable: existing files are kept."""
    stats = stats or HarvestStats()
    site_id = site.get("id") or site.get("name") or f"{site['lat']:.5f}_{site['lon']:.5f}"
    safe_id = "".join(c for c in str(site_id) if c.isalnum() or c in "-_")[:80]
    site_dir = area_dir(area) / "images" / safe_id
    site_dir.mkdir(parents=True, exist_ok=True)

    found = discover_files(client, site["lat"], site["lon"], radius_m, per_site)
    stats.discovered += len(found)
    if not found:
        return []
    time.sleep(API_PAUSE)

    wanted = found[:per_site]
    pending = [f for f in wanted if not _image_path(site_dir, f["title"]).exists()]
    urls = resolve_thumbs(client, [f["title"] for f in pending]) if pending else {}

    records: list[dict[str, Any]] = []
    for f in wanted:
        dest = _image_path(site_dir, f["title"])
        if not dest.exists():
            url = urls.get(f["title"])
            if not url:
                stats.failed += 1
                continue
            try:
                time.sleep(DOWNLOAD_PAUSE)
                img = client.get(url)
                if img.status_code != 200 or len(img.content) < 3000:
                    stats.failed += 1
                    continue
                dest.write_bytes(img.content)
                stats.downloaded += 1
            except Exception:
                stats.failed += 1
                continue
        else:
            stats.skipped += 1
        records.append({
            "site_id": str(site_id), "site_name": site.get("name", ""),
            "site_lat": site["lat"], "site_lon": site["lon"],
            "title": f["title"], "path": str(dest),
            "img_lat": f.get("lat"), "img_lon": f.get("lon"),
            "dist_m": f.get("dist_m"),
        })
    return records


def harvest(area: str, sites: list[dict[str, Any]], *, per_site: int = 60,
            radius_m: int = 500, progress=None) -> tuple[list[dict[str, Any]], HarvestStats]:
    """Harvest reference imagery for a list of sites into one named area."""
    if not SETTINGS.net_allowed():
        raise OfflineError("Corpus building needs network access; offline mode is on.")

    stats = HarvestStats(sites=len(sites))
    manifest_path = area_dir(area) / "manifest.json"
    records: list[dict[str, Any]] = []
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            records = json.loads(manifest_path.read_text())

    known = {(r["site_id"], r["title"]) for r in records}
    with _client() as client:
        for i, site in enumerate(sites, 1):
            if progress:
                progress(f"[{i}/{len(sites)}] {site.get('name') or site['site_id'] if 'site_id' in site else site.get('name','site')}")
            try:
                fresh = harvest_site(client, area, site, per_site=per_site,
                                     radius_m=radius_m, stats=stats)
            except Exception as exc:
                stats.notes.append(f"{site.get('name','site')}: {type(exc).__name__}")
                continue
            for rec in fresh:
                if (rec["site_id"], rec["title"]) not in known:
                    known.add((rec["site_id"], rec["title"]))
                    records.append(rec)
            # Written every site so an interrupted harvest loses nothing.
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(records, indent=1))
    return records, stats


def load_manifest(area: str) -> list[dict[str, Any]]:
    path = area_dir(area) / "manifest.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def list_areas() -> Iterator[tuple[str, int]]:
    root = corpus_root()
    if not root.exists():
        return
    for d in sorted(root.iterdir()):
        if d.is_dir():
            yield d.name, len(load_manifest(d.name))
