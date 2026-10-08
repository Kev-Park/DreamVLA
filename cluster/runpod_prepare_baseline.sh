#!/bin/bash
set -euo pipefail
ROOT=/workspace/kevin
DS=$ROOT/datasets/pick_effective1200/ds
test ! -e "$DS/READY" || exit 0
"$ROOT/Isaac-GR00T/.venv/bin/python" "$ROOT/DreamVLA_runtime/cluster/build_effective_dataset.py" \
  "$ROOT/datasets/af60v8f_base/lerobot/ds" "$DS" --count 1200 --seed 42
