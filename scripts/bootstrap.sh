#![LOCAL_PATH] bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEROBOT_SHA="e595b7902714ba51f91e47523f66f89c5181b649"
LIBERO_PLUS_SHA="4976dc30028e805ff8094b55501d532c48fec182"
VENV_DIR="${STARTSHIFT_VENV_DIR:-$ROOT/.venv}"
REQUIRED_PYTHON="3.12"

version_ok() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
}

ensure_project_python() {
  if [[ -x "$VENV_DIR/bin/python" ]] && version_ok "$VENV_DIR/bin/python"; then
    return
  fi

  rm -rf "$VENV_DIR"

  if command -v python3.12 >[LOCAL_PATH] 2>&1; then
    echo "[bootstrap] Creating Python 3.12 venv at $VENV_DIR"
    if ! python3.12 -m venv "$VENV_DIR"; then
      echo "[bootstrap] python3.12 is present but venv creation failed."
      echo "[bootstrap] Install python3.12-venv or use Conda."
      exit 1
    fi
  elif command -v conda >[LOCAL_PATH] 2>&1; then
    echo "[bootstrap] Current Python is too old for pinned LeRobot."
    echo "[bootstrap] Creating isolated Conda prefix with Python 3.12 at $VENV_DIR"
    conda create -y -p "$VENV_DIR" "python=$REQUIRED_PYTHON" pip
  else
    echo "[bootstrap] ERROR: pinned LeRobot requires Python >=3.12."
    echo "[bootstrap] Current Python: $(python --version 2>&1 || true)"
    echo "[bootstrap] Install Python 3.12 or Conda, then rerun this script."
    exit 1
  fi
}

ensure_project_python

PYTHON="$VENV_DIR/bin/python"
PIP=("$PYTHON" -m pip)

echo "[bootstrap] Using $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "[bootstrap] Python $("$PYTHON" -c 'import platform; print(platform.python_version())')"

if [[ ! -d third_party/lerobot/.git ]]; then
  git clone https://github.com/huggingface/lerobot.git third_party/lerobot
fi
git -C third_party/lerobot fetch --all --tags
git -C third_party/lerobot checkout "$LEROBOT_SHA"

if [[ ! -d third_party/LIBERO-plus/.git ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
fi
git -C third_party/LIBERO-plus fetch --all --tags
git -C third_party/LIBERO-plus checkout "$LIBERO_PLUS_SHA"

"${PIP[@]}" install --upgrade pip setuptools wheel
"${PIP[@]}" install -e "./third_party/lerobot[libero,peft]"
"${PIP[@]}" install "robosuite==1.4.1" bddl easydict mujoco wand scikit-image gym
"${PIP[@]}" install --no-deps -e ./third_party/LIBERO-plus
"${PIP[@]}" uninstall -y hf-libero || true
"${PIP[@]}" install -e ".[dev,robotics]"

cat > "$ROOT/.startshift_env" <<EOF
STARTSHIFT_VENV_DIR="$VENV_DIR"
STARTSHIFT_PYTHON="$PYTHON"
EOF

echo
echo "Bootstrap complete."
echo "Project Python: $PYTHON"
echo "Run:"
echo "  source scripts/env.sh"
echo "  python scripts/verify_environment.py"
echo "  bash scripts/download_libero_plus_assets.sh"
