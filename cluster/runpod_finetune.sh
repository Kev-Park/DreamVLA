#!/bin/bash
set -euo pipefail
DS=${1:?dataset}; RUN=${2:?run}; STEPS=${3:?steps}
export HF_HOME=/workspace/hf CUDA_HOME=/usr/local/cuda
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd ~/kevin/Isaac-GR00T
mkdir -p ~/kevin/checkpoints /workspace/logs
# Keep the durable dataset intact; train from a verified local-disk copy.
LOCAL_DS=/opt/training_data/$RUN
mkdir -p "$LOCAL_DS"
(cd "$DS"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "/workspace/logs/$RUN.dataset.sha256"
cp -a "$DS/." "$LOCAL_DS/"
(cd "$LOCAL_DS"; sha256sum --quiet -c "/workspace/logs/$RUN.dataset.sha256")
DS=$LOCAL_DS
.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic --num-gpus 1 --global-batch-size 16 \
  --dataloader-num-workers 8 --output-dir ~/kevin/checkpoints \
  --experiment-name "$RUN" --max-steps "$STEPS" --save-steps 10000 \
  --save-total-limit 2 --save-only-model
