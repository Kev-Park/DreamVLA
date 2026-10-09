#!/bin/bash
set -euo pipefail
DS=${1:?dataset}; RUN=${2:?run}; STEPS=${3:?steps}
BATCH=${4:-$(python3 -c 'import json,pathlib,sys; p=pathlib.Path("/workspace/queue/current.json"); j=json.loads(p.read_text()) if p.exists() else {}; print(j.get("global_batch_size",16) if j.get("run")==sys.argv[1] else 16)' "$RUN")}
case "$BATCH" in 16|32|64|128) ;; *) echo "Unsupported experiment batch size: $BATCH"; exit 1;; esac
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
GPUS=1
test ! -f /workspace/gpu_count || GPUS=$(cat /workspace/gpu_count)
case "$GPUS" in 1|2|4|8) ;; *) echo "Invalid GPU count"; exit 1;; esac
LAUNCH=(.venv/bin/python)
if [ "$GPUS" -gt 1 ]; then
  # The copied single-GPU environment may omit this declared dependency.
  .venv/bin/python -c 'import deepspeed' || uv pip install --python .venv/bin/python deepspeed==0.17.6
  LAUNCH+=(-m torch.distributed.run --standalone --nproc_per_node "$GPUS")
fi
ENTRY=(gr00t/experiment/launch_finetune.py)
RESUME=$(python3 -c 'import json,pathlib,sys; p=pathlib.Path("/workspace/queue/current.json"); j=json.loads(p.read_text()) if p.exists() else {}; print(j.get("resume_checkpoint","") if j.get("run")==sys.argv[1] else "")' "$RUN")
if [ -n "$RESUME" ]; then
  ENTRY=("$(dirname "$(readlink -f "$0")")/recover_weights.py" --checkpoint "$RESUME" --)
fi
"${LAUNCH[@]}" "${ENTRY[@]}" \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic --num-gpus "$GPUS" --global-batch-size "$BATCH" \
  --dataloader-num-workers 8 --output-dir ~/kevin/checkpoints \
  --experiment-name "$RUN" --max-steps "$STEPS" --save-steps 10000 \
  --save-total-limit 2 --save-only-model
