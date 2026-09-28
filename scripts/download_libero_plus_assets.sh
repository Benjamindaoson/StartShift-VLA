#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

if ! command -v hf >/dev/null 2>&1; then
  echo "[assets] ERROR: Hugging Face CLI not found in the project environment." >&2
  echo "[assets] Run: bash scripts/bootstrap.sh && source scripts/env.sh" >&2
  exit 1
fi

TMP_DIR="${TMPDIR:-/tmp}/startshift-libero-plus-assets"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "$TMP_DIR"
PKG_DIR="$(python -c 'import pathlib, libero; print(pathlib.Path(libero.__file__).resolve().parent)')"
unzip -o "$TMP_DIR/assets.zip" -d "$PKG_DIR"
echo "LIBERO-Plus assets installed under $PKG_DIR"
