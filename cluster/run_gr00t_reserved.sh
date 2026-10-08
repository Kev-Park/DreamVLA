#!/bin/bash
# Reserve the full fine-tune footprint before CUDA/model initialization.
set -u
DS=${1:?}; RUN=${2:?}; STEPS=${3:?}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
GPU=$(python3 ~/kevin/wt/dagger-watch-sparse/cluster/gpu_slots.py reserve --owner $$ --count 1 --mb 45000)
[ -n "$GPU" ] || { echo "No GPU capacity after startup reservations; Adroit remains the fallback"; exit 2; }
trap 'python3 ~/kevin/wt/dagger-watch-sparse/cluster/gpu_slots.py release --owner $$ --gpu "$GPU"' EXIT
cd ~/kevin/Isaac-GR00T || exit 1
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" --embodiment-tag unitree_g1_sonic \
  --num-gpus 1 --global-batch-size 16 --dataloader-num-workers 8 --output-dir ~/kevin/checkpoints \
  --experiment-name "$RUN" --max-steps "$STEPS" --save-steps "$STEPS" --save-total-limit 1 --save-only-model
