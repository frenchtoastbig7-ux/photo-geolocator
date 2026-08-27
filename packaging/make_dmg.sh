#!/bin/bash
# Build a compressed drag-to-Applications .dmg from the built app bundle.
#
#   bash packaging/make_dmg.sh [dist/Geolocation Workbench.app] [out.dmg]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$HERE/dist/Geolocation Workbench.app}"
OUT="${2:-$HERE/dist/GeolocationWorkbench.dmg}"
VOL="Geolocation Workbench"

[ -d "$APP" ] || { echo "no app bundle at $APP — run build_app.sh first"; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"      # drag target

# A short note in the volume, since the app is unsigned by an identified
# developer and first launch needs right-click → Open.
cat > "$STAGE/READ ME FIRST.txt" <<'NOTE'
Geolocation Workbench
=====================

Install
  Drag "Geolocation Workbench" onto the Applications folder shown here.

First launch
  macOS blocks apps from unidentified developers. The first time only:
  right-click (or Control-click) the app in Applications and choose "Open",
  then confirm. Afterwards it opens normally by double-clicking.

What it does
  Analyses a single photograph for location clues and shows ranked
  candidates on a map. Everything runs on this Mac. The image is never
  uploaded anywhere.

Optional scene model
  Analysis Options > "Install scene model" fetches a 1.7 GB neural model
  for coarse country estimation. It is optional: metadata forensics, OCR,
  solar geometry and evidence fusion all work without it.

Your data
  Cases and any downloaded model live in
  ~/Library/Application Support/Geolocation Workbench
  Delete that folder to remove everything the app has stored.
NOTE

rm -f "$OUT"
mkdir -p "$(dirname "$OUT")"
hdiutil create -volname "$VOL" -srcfolder "$STAGE" -fs HFS+ \
  -format UDZO -imagekey zlib-level=9 -ov "$OUT" >/dev/null

echo "wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
