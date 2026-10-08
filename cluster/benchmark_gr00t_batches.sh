#!/bin/bash
set -uo pipefail
PID=${1:?production PID}
export XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HOME=/workspace/hf
export CUDA_HOME=/usr/local/cuda WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd /opt/kevin/Isaac-GR00T
for batch in 32 64 128; do
  timeout --signal=TERM --kill-after=15 420 .venv/bin/python \
    /workspace/kevin/DreamVLA_runtime/cluster/benchmark_gr00t_batch.py \
    --config /workspace/kevin/checkpoints/af60v8f57_dagger2_run01/experiment_cfg/config.yaml \
    --batch "$batch" --output "/workspace/bench_batch_$batch" \
    --production-pid "$PID" > "/workspace/logs/bench_batch_$batch.log" 2>&1
done
