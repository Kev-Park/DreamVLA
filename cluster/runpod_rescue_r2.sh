#!/bin/bash
# Run on Bluesclues: rescue a stopped cloud run without touching live training.
set -euo pipefail
SRC_HOST=${1:?}; SRC_PORT=${2:?}; DST_HOST=${3:?}; DST_PORT=${4:?}
RUN=af60v8f57_dagger2_run01
KEY=$HOME/kevin/runpod_transport/key
SRC=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p "$SRC_PORT" "root@$SRC_HOST")
DST=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p "$DST_PORT" "root@$DST_HOST")
SSH_SRC="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $SRC_PORT"
SSH_DST="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $DST_PORT"
BACKUP=$HOME/kevin/runpod_transport/recovered_r2/checkpoint-90000
mkdir -p "$BACKUP"
"${DST[@]}" 'mkdir -p /opt/kevin/Isaac-GR00T/.venv /workspace/hf /workspace/logs /workspace/queue /workspace/kevin/checkpoints/af60v8f57_dagger2_run01/checkpoint-90000 /workspace/kevin/datasets/af60v8f_ds_dagger2/lerobot/ds; git clone --depth 1 --branch g1 https://github.com/Kev-Park/DreamVLA.git /workspace/kevin/DreamVLA_runtime'
# Preserve an independent copy before resuming on the new Pod.
"${SRC[@]}" 'cd /workspace/kevin/checkpoints/af60v8f57_dagger2_run01/checkpoint-90000; find . -type f -print0 | sort -z | xargs -0 sha256sum' > "$BACKUP/../checkpoint.sha256"
rsync -a --partial -e "$SSH_SRC" "root@$SRC_HOST:/workspace/kevin/checkpoints/$RUN/checkpoint-90000/" "$BACKUP/" &
P_CK=$!
rsync -az --partial --compress-choice=zstd --compress-level=1 -e "$SSH_DST" "$HOME/kevin/Isaac-GR00T/.venv/" "root@$DST_HOST:/opt/kevin/Isaac-GR00T/.venv/" &
P_ENV=$!
rsync -a --partial -e "$SSH_DST" "$HOME/kevin/datasets/af60v8f_ds_dagger2/lerobot/ds/" "root@$DST_HOST:/workspace/kevin/datasets/af60v8f_ds_dagger2/lerobot/ds/" &
P_DS=$!
"${SRC[@]}" 'tar -C /workspace/hf -cf - .' | "${DST[@]}" 'tar --no-same-owner -C /workspace/hf -xf -' &
P_HF=$!
wait "$P_CK"
(cd "$BACKUP"; sha256sum --quiet -c ../checkpoint.sha256)
touch "$BACKUP/../backup.verified"
rsync -a --partial -e "$SSH_DST" "$BACKUP/" "root@$DST_HOST:/workspace/kevin/checkpoints/$RUN/checkpoint-90000/"
"${DST[@]}" 'cat > /workspace/logs/r2_checkpoint.sha256' < "$BACKUP/../checkpoint.sha256"
"${DST[@]}" 'cd /workspace/kevin/checkpoints/af60v8f57_dagger2_run01/checkpoint-90000; sha256sum --quiet -c /workspace/logs/r2_checkpoint.sha256'
wait "$P_ENV"; wait "$P_DS"; wait "$P_HF"
touch "$BACKUP/../all_transfers.done"
"${DST[@]}" 'bash /workspace/kevin/DreamVLA_runtime/cluster/runpod_prepare.sh && touch /workspace/logs/recovery_environment.ready'
"${DST[@]}" 'python3 -c '\''import json; from pathlib import Path; j=dict(run="af60v8f57_dagger2_run01",steps=127000,line="af60v8f57",stage="epoch57",global_batch_size=16,dataset="/workspace/kevin/datasets/af60v8f_ds_dagger2/lerobot/ds",state="recovering"); Path("/workspace/queue/current.json").write_text(json.dumps(j))'\'''
"${DST[@]}" 'nohup bash /workspace/kevin/DreamVLA_runtime/cluster/runpod_recover.sh > /workspace/logs/recovery.log 2>&1 < /dev/null &'
echo RECOVERY_LAUNCHED
