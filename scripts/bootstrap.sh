#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEROBOT_SHA="e595b7902714ba51f91e47523f66f89c5181b649"
LIBERO_PLUS_SHA="4976dc30028e805ff8094b55501d532c48fec182"

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

python -m pip install --upgrade pip
python -m pip install -e "./third_party/lerobot[libero,peft]"
python -m pip install "robosuite==1.4.1" bddl easydict mujoco wand scikit-image gym
python -m pip install --no-deps -e ./third_party/LIBERO-plus
python -m pip uninstall -y hf-libero || true
python -m pip install -e ".[dev,robotics]"

echo
echo "Bootstrap complete."
echo "Source scripts/env.sh before LIBERO-Plus runs."
echo "Then run scripts/download_libero_plus_assets.sh once."
