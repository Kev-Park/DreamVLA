#!/bin/bash
# Established 2x5090 ZeRO-2 recipe: per-device batch 8, effective batch 16.
set -eo pipefail
DS=${1:?dataset}; RUN=${2:?run}; STEPS=${3:?steps}
WORK=$HOME/kevin/dline/$RUN
mkdir -p "$WORK"
[ ! -e "$WORK/exit_code.txt" ]
trap 'rc=$?; printf "%s\n" "$rc" > "$WORK/exit_code.txt"' EXIT
echo $$ > "$WORK/worker.pid"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
# This is the previously validated transport on biped; P2P-only disable is insufficient.
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_HOME=/usr/local/cuda
export TOKENIZERS_PARALLELISM=false
ARGS=()
# Intermediate models remain inference-complete; retain final-round optimizer state.
[[ "$RUN" == *_r3 ]] || ARGS+=(--save-only-model)
cd "$HOME/kevin/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic --num-gpus 2 --global-batch-size 16 \
  --gradient-accumulation-steps 1 --learning-rate 0.0001 --dataloader-num-workers 4 \
  --output-dir "$HOME/kevin/checkpoints" --experiment-name "$RUN" \
  --max-steps "$STEPS" --save-steps "$STEPS" --save-total-limit 1 "${ARGS[@]}"
