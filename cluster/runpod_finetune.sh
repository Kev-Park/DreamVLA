#!/bin/bash
set -euo pipefail
DS=${1:?dataset}; RUN=${2:?run}; STEPS=${3:?steps}
export HF_HOME=/workspace/hf CUDA_HOME=/usr/local/cuda
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd ~/kevin/Isaac-GR00T
mkdir -p ~/kevin/checkpoints /workspace/logs
.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic --num-gpus 1 --global-batch-size 16 \
  --dataloader-num-workers 8 --output-dir ~/kevin/checkpoints \
  --experiment-name "$RUN" --max-steps "$STEPS" --save-steps 1000 \
  --save-total-limit 2 --save-only-model
