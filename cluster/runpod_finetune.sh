#!/bin/bash
set -euo pipefail
DS=${1:?dataset}; RUN=${2:?run}; STEPS=${3:?steps}
export HF_HOME=/workspace/hf CUDA_HOME=/usr/local/cuda
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=8
cd ~/kevin/Isaac-GR00T
mkdir -p ~/kevin/checkpoints /workspace/logs
# Keep the durable dataset intact; train from a verified local-disk copy.
LOCAL_DS=/opt/training_data/$RUN
mkdir -p "$LOCAL_DS"
(cd "$DS"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "/workspace/logs/$RUN.dataset.sha256"
cp -a "$DS/." "$LOCAL_DS/"
(cd "$LOCAL_DS"; sha256sum --quiet -c "/workspace/logs/$RUN.dataset.sha256")
DS=$LOCAL_DS
GPUS=1
test ! -f /workspace/gpu_count || GPUS=$(cat /workspace/gpu_count)
case "$GPUS" in 1|2|4|8) ;; *) echo "Invalid GPU count"; exit 1;; esac
LAUNCH=(.venv/bin/python)
if [ "$GPUS" -gt 1 ]; then
  # The copied single-GPU environment may omit this declared dependency.
  .venv/bin/python -c 'import deepspeed' || uv pip install --python .venv/bin/python deepspeed==0.17.6
  LAUNCH+=(-m torch.distributed.run --standalone --nproc_per_node "$GPUS")
fi
"${LAUNCH[@]}" ~/kevin/DreamVLA_runtime/cluster/runpod_train.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic --num-gpus "$GPUS" --global-batch-size 16 \
  --dataloader-num-workers 8 --output-dir ~/kevin/checkpoints \
  --experiment-name "$RUN" --max-steps "$STEPS" --save-steps 10000 \
  --save-total-limit 2 --save-only-model
