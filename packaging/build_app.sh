#!/bin/bash
# Build dist/"Geolocation Workbench.app" — a self-contained macOS bundle.
#
#   bash packaging/build_app.sh
#
# Needs a working dev environment (see README). The bundle carries Python,
# PyTorch and every dependency; StreetCLIP's weights are downloaded by the
# app on request, not bundled.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HERE/dist/Geolocation Workbench.app"
PY="${PYTHON:-$HERE/.venv/bin/python}"

# PYTHON may be a path or a bare command name. CI passes `PYTHON=python`, and
# `[ -x python ]` tests for a file called "python" in the working directory,
# which failed every CI build before PyInstaller ever ran. Resolve names on
# PATH first, then check the result.
if [[ "$PY" != */* ]]; then
  PY="$(command -v "$PY" || true)"
fi
[ -n "$PY" ] && [ -x "$PY" ] || { echo "no interpreter found (PYTHON=${PYTHON:-unset}) — create one with 'uv venv --python 3.12'"; exit 1; }

echo "==> regenerating icon"
"$PY" "$HERE/packaging/make_icon.py"

echo "==> running PyInstaller (this takes a few minutes)"
rm -rf "$HERE/build" "$HERE/dist"
"$PY" -m PyInstaller "$HERE/packaging/geoloc-mac.spec" --noconfirm --log-level WARN

[ -d "$APP" ] || { echo "build produced no bundle at $APP"; exit 1; }

# An ad-hoc signature keeps the bundle launchable and stops macOS complaining
# that it is damaged. It is NOT notarised: users still get Gatekeeper's
# unidentified-developer prompt on first open.
echo "==> ad-hoc signing"
codesign --force --deep --sign - "$APP" 2>/dev/null || \
  echo "    (codesign unavailable — bundle will still run after right-click → Open)"

# Locally built bundles inherit the quarantine flag from downloaded wheels.
xattr -cr "$APP" 2>/dev/null || true

echo "==> done: $APP  ($(du -sh "$APP" | cut -f1))"
