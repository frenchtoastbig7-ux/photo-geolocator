# PyInstaller spec — builds "Geolocation Workbench.app" for macOS.
#
#   pyinstaller packaging/geoloc-mac.spec --noconfirm
#
# The bundle carries the interpreter, PyTorch and every dependency, so it
# runs on a Mac with no Python, Homebrew or pip. StreetCLIP's 1.7 GB weights
# are deliberately NOT bundled: the app installs them on request into
# ~/Library/Application Support, and works fully without them.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (injected)
VERSION = "0.1.0"

hiddenimports: list[str] = []

# Server internals resolved dynamically at runtime, which static analysis misses.
for pkg in ("uvicorn", "anyio", "pydantic", "pydantic_core", "websockets",
            "httptools", "uvloop", "httpx", "httpcore", "h11"):
    hiddenimports += collect_submodules(pkg)

# Starlette looks up python-multipart under both spellings when parsing forms.
hiddenimports += ["multipart", "python_multipart"]

# pyobjc: Vision drives OCR and face detection; WebKit/AppKit drive the window.
for pkg in ("objc", "Foundation", "AppKit", "WebKit", "Quartz", "Vision",
            "CoreFoundation", "PyObjCTools"):
    hiddenimports += collect_submodules(pkg)
hiddenimports += collect_submodules("webview")

# huggingface_hub is imported lazily by the model installer.
hiddenimports += collect_submodules("huggingface_hub")

# torch.backends is resolved lazily by attribute access (torch.backends.mps),
# which PyInstaller cannot see. Without this the device probe raises
# AttributeError inside the bundle and every request to /api/settings 500s.
hiddenimports += collect_submodules("torch.backends")
hiddenimports += ["torch.backends.mps", "torch.backends.cpu",
                  "torch.backends.mkldnn", "torch.backends.quantized"]
hiddenimports += ["geoloc.server", "geoloc.pipeline", "geoloc.modelmgr",
                  "scipy.spatial", "scipy.special"]

# The place-recognition stack is imported inside functions so that a build
# without it still runs, which also means PyInstaller's static analysis never
# sees it. Named explicitly or the frozen app silently loses visual matching.
hiddenimports += collect_submodules("geoloc.vpr")
hiddenimports += ["geoloc.metaverify", "geoloc.analyzers.shadows",
                  "geoloc.analyzers.heading", "geoloc.analyzers.solar",
                  "geoloc.geo.facility", "geoloc.geo.textgeo",
                  "geoloc.vpr.model", "geoloc.vpr.corpus",
                  "geoloc.vpr.index", "geoloc.vpr.match",
                  "geoloc.vpr.cli_commands",
                  "torchvision.transforms", "torchvision.models"]

datas = [
    (os.path.join(REPO, "src", "geoloc", "web"), "geoloc/web"),
]
datas += collect_data_files("webview")

# geonamescache ships four city datasets (~190 MB). grid.py pins the 15000
# threshold, so only cities15000.json is ever read -- carrying the rest would
# roughly double the download for no functional gain.
for src, dest in collect_data_files("geonamescache"):
    base = os.path.basename(src)
    if base.startswith("cities") and base != "cities15000.json":
        continue
    datas.append((src, dest))

# Only whole packages that nothing in the dependency graph imports.
#
# Do NOT exclude submodules of torch, torchvision, scipy or sympy. Their
# internal imports are not what the names suggest: excluding `torch.testing`
# looks obviously safe and is not -- `torch.autograd.gradcheck` imports it
# unconditionally, so the exclusion made `import torch` fail outright and
# silently reduced the app to CPU-only with no scene model. The saving was
# tens of megabytes; the cost was the entire ML feature set.
excludes = [
    "tkinter", "matplotlib", "IPython", "notebook", "jupyter",
    "pytest", "_pytest",
]

a = Analysis(
    [os.path.join(REPO, "src", "geoloc", "desktop.py")],
    pathex=[os.path.join(REPO, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Geolocation Workbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                       # windowed app, no terminal
    icon=os.path.join(REPO, "packaging", "geoloc.icns"),
    disable_windowed_traceback=False,
    target_arch=None,                    # build for the host architecture
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="Geolocation Workbench",
)
app = BUNDLE(
    coll,
    name="Geolocation Workbench.app",
    icon=os.path.join(REPO, "packaging", "geoloc.icns"),
    bundle_identifier="local.geoloc.workbench",
    version=VERSION,
    info_plist={
        "CFBundleName": "Geolocation Workbench",
        "CFBundleDisplayName": "Geolocation Workbench",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        # A regular Dock app with a window, not a menu-bar agent.
        "LSUIElement": False,
        "NSRequiresAquaSystemAppearance": False,
        "NSHumanReadableCopyright": "MIT licensed. Analyses run locally.",
    },
)
