#![LOCAL_PATH] bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

if ! command -v hf >[LOCAL_PATH] 2>&1; then
  echo "[assets] ERROR: Hugging Face CLI not found in the project environment." >&2
  echo "[assets] Run: bash scripts/bootstrap.sh && source scripts/env.sh" >&2
  exit 1
fi

TMP_DIR="${TMPDIR:-/tmp}/startshift-libero-plus-assets"
mkdir -p "$TMP_DIR"

if [[ ! -s "$TMP_DIR/assets.zip" ]]; then
  hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "$TMP_DIR"
else
  echo "[assets] Reusing $TMP_DIR/assets.zip"
fi
PKG_DIR="$ROOT/third_party/LIBERO-plus/libero/libero"
DATA_ROOT="${STARTSHIFT_LIBERO_DATA_ROOT:-$PKG_DIR}"
CONFIG_ROOT="${LIBERO_CONFIG_PATH:-$ROOT/.libero}"
mkdir -p "$DATA_ROOT" "$CONFIG_ROOT"

EXTRACT_DIR="$TMP_DIR/extract"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q "$TMP_DIR/assets.zip" -d "$EXTRACT_DIR"
ASSETS_DIR="$(find "$EXTRACT_DIR" -type d -name assets -print -quit)"
if [[ -z "$ASSETS_DIR" ]]; then
  echo "[assets] ERROR: assets directory not found in archive." >&2
  exit 1
fi
if [[ ! -e "$DATA_ROOT/assets" ]]; then
  mv "$ASSETS_DIR" "$DATA_ROOT/assets"
fi

export STARTSHIFT_LIBERO_PKG_DIR="$PKG_DIR"
export STARTSHIFT_LIBERO_DATA_ROOT="$DATA_ROOT"
export STARTSHIFT_LIBERO_CONFIG_FILE="$CONFIG_ROOT/config.yaml"
python - <<'PY'
import os
from pathlib import Path

import yaml

package = Path(os.environ["STARTSHIFT_LIBERO_PKG_DIR"]).resolve()
data_root = Path(os.environ["STARTSHIFT_LIBERO_DATA_ROOT"]).resolve()
config_file = Path(os.environ["STARTSHIFT_LIBERO_CONFIG_FILE"]).resolve()
config = {
    "benchmark_root": str(package),
    "bddl_files": str(package / "bddl_files"),
    "init_states": str(package / "init_files"),
    "datasets": str(data_root / "datasets"),
    "assets": str(data_root / "assets"),
}
config_file.write_text(yaml.safe_dump(config, sort_keys=True))
print(f"LIBERO config: {config_file}")
print(f"LIBERO-Plus assets: {config['assets']}")
PY
