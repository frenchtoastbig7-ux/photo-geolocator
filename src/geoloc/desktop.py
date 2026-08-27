"""Native desktop entry point.

This is what `Geolocation Workbench.app` runs. It starts the FastAPI server
in-process on a random loopback port, waits for it to answer, then puts the
workbench in a real WKWebView window rather than sending the analyst to a
browser tab.

Three things here are consequences of being frozen into a .app rather than
run from a checkout:

* A PyInstaller binary cannot re-exec ``python -m uvicorn``, so the server
  runs on a daemon thread inside this process.
* The bundle is read-only, so case data, logs and model weights are
  redirected to ~/Library/Application Support before anything imports the
  config module and caches those paths.
* The port is chosen at random. A fixed port would collide with a second
  copy of the app, or with a developer running `geoloc serve`.
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

APP_NAME = "Geolocation Workbench"


def _app_support() -> Path:
    path = Path.home() / "Library" / "Application Support" / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


SUPPORT = _app_support()
LOG_PATH = SUPPORT / "workbench.log"

# Must precede any geoloc import: config resolves these at import time.
os.environ.setdefault("GEOLOC_CASE_DIR", str(SUPPORT / "cases"))
os.environ.setdefault("GEOLOC_MODEL_CACHE", str(SUPPORT / "models"))
# Keep HuggingFace's own caches inside the app's directory too, so
# uninstalling means deleting one folder.
os.environ.setdefault("HF_HOME", str(SUPPORT / "models" / "huggingface"))


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with contextlib.suppress(OSError), LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


PORT = int(os.environ.get("GEOLOC_PORT") or _free_port())
URL = f"http://127.0.0.1:{PORT}/"


def _serve() -> None:
    """Run uvicorn in-process. Imports stay local so they happen off the
    main thread, while PyInstaller's static analysis still sees them."""
    try:
        import uvicorn

        from geoloc.server import app

        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
    except Exception:
        _log("server thread crashed:\n" + traceback.format_exc())


def _wait_ready(timeout: float = 90.0) -> bool:
    """Poll until the server returns a well-formed settings document.

    Checking for a parseable 200 rather than merely "did not raise" matters:
    a bundle missing a dependency answers 500 on every request, and a probe
    that tolerates that reports a broken app as ready.
    """
    import json

    deadline = time.time() + timeout
    probe = URL + "api/settings"
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe, timeout=3) as r:
                if r.status == 200:
                    payload = json.loads(r.read().decode("utf-8"))
                    if "device" in payload:
                        return True
                    last = f"unexpected payload: {sorted(payload)[:5]}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.3)
    if last:
        _log(f"readiness probe never succeeded — last failure: {last}")
    return False


def _alert(title: str, message: str) -> None:
    script = (f'display alert "{title}" message "{message}" as critical')
    with contextlib.suppress(Exception):
        subprocess.run(["osascript", "-e", script], check=False, timeout=60)


def _run_window() -> None:
    """Show the workbench in a native WKWebView window."""
    import webview

    window = webview.create_window(
        APP_NAME,
        URL,
        width=1440,
        height=900,
        min_size=(1024, 680),
        text_select=True,
        confirm_close=False,
    )

    def _on_closed() -> None:
        # The server runs on a daemon thread; leaving the process alive after
        # the last window closes would strand it with no way to reach it.
        os._exit(0)

    window.events.closed += _on_closed
    webview.start(private_mode=False, storage_path=str(SUPPORT / "webview"))


def _probe(path: str) -> dict:
    import json

    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _smoke_test() -> int:
    """Verify the bundle actually works, then exit.

    Checks capability rather than mere responsiveness. A PyInstaller bundle
    can serve requests perfectly while a mis-scoped exclusion has broken
    PyTorch, leaving the app quietly degraded; asserting on the reported
    device and torch availability turns that into a build failure.
    """
    try:
        settings = _probe("api/settings")
        model = _probe("api/model")
    except Exception as exc:
        _log(f"smoke test could not query the API: {exc}")
        return 1

    device = settings.get("device", "?")
    torch_ok = bool(model.get("torch_available"))
    print(f"SMOKE device={device} torch={torch_ok} "
          f"scene_model_installed={bool(model.get('installed'))}", flush=True)
    print(f"{APP_NAME} ready at {URL}", flush=True)

    if getattr(sys, "frozen", False) and not torch_ok:
        _log("smoke test FAILED: PyTorch is bundled but not importable")
        print("SMOKE FAIL: PyTorch is bundled but not importable — check the "
              "spec's `excludes` for over-broad submodule exclusions",
              flush=True)
        return 1

    _log(f"smoke test passed (device={device}, torch={torch_ok})")
    return 0


def main() -> int:
    _log(f"starting {APP_NAME} on {URL} (frozen={getattr(sys, 'frozen', False)})")
    threading.Thread(target=_serve, daemon=True, name="uvicorn").start()

    if not _wait_ready():
        _log("server did not become ready")
        _alert(APP_NAME,
               "The analysis server failed to start. "
               f"See the log at {LOG_PATH}")
        return 1

    _log("server ready")

    # Two windowless modes, kept distinct because they need opposite
    # lifetimes. Creating an NSApplication on a machine with no window server
    # aborts the process uncatchably, so CI can never open the window.
    #
    #   GEOLOC_SMOKE_TEST : confirm the server works, then exit 0.
    #   GEOLOC_HEADLESS   : serve until interrupted, with no window.
    if os.environ.get("GEOLOC_SMOKE_TEST"):
        return _smoke_test()

    if os.environ.get("GEOLOC_HEADLESS"):
        _log("headless mode: serving, no window")
        print(f"{APP_NAME} ready at {URL}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _log("interrupted")
        return 0

    try:
        _run_window()
    except Exception:
        _log("webview failed, falling back to the default browser:\n"
             + traceback.format_exc())
        import webbrowser
        webbrowser.open(URL)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
