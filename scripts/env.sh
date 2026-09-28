#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/third_party/LIBERO-plus:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
