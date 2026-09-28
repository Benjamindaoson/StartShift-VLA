#!/usr/bin/env bash
set -euo pipefail

TMP_DIR="${TMPDIR:-/tmp}/startshift-libero-plus-assets"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "$TMP_DIR"
PKG_DIR="$(python -c 'import pathlib, libero; print(pathlib.Path(libero.__file__).resolve().parent)')"
unzip -o "$TMP_DIR/assets.zip" -d "$PKG_DIR"
echo "LIBERO-Plus assets installed under $PKG_DIR"
