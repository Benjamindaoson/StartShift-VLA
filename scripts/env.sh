#![LOCAL_PATH] bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="${STARTSHIFT_VENV_DIR:-$ROOT/.venv}"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  export STARTSHIFT_VENV_DIR="$VENV_DIR"
  export STARTSHIFT_PYTHON="$VENV_DIR/bin/python"
  export PATH="$VENV_DIR/bin:$PATH"
elif [[ -f "$ROOT/.startshift_env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.startshift_env"
  export STARTSHIFT_VENV_DIR
  export STARTSHIFT_PYTHON
  export PATH="$(dirname "$STARTSHIFT_PYTHON"):$PATH"
else
  echo "[env] WARNING: project Python environment not found. Run: bash scripts/bootstrap.sh" >&2
fi

export PYTHONPATH="$ROOT/third_party/LIBERO-plus:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/.libero}"
