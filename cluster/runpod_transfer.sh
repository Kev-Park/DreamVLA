#!/bin/bash
# Run on bluesclues: stream existing inputs directly to the paid worker.
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}
DEST="root@$HOST"
SSH=(ssh -i "$HOME/kevin/runpod_transport/key" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$PORT")
"${SSH[@]}" "$DEST" 'mkdir -p /workspace/kevin /workspace/hf/hub /workspace/logs'
(
  tar -C ~/kevin -cf - Isaac-GR00T/.venv |
    "${SSH[@]}" "$DEST" 'mkdir -p /opt/kevin; tar --no-same-owner -C /opt/kevin -xf - && touch /workspace/logs/environment.ready'
) & envpid=$!
(
  tar -C ~/.cache/huggingface/hub -cf - models--nvidia--GR00T-N1.7-3B models--nvidia--Cosmos-Reason2-2B |
    "${SSH[@]}" "$DEST" 'tar --no-same-owner -C /workspace/hf/hub -xf - && touch /workspace/logs/models.ready'
) & modelpid=$!
(
  tar -C ~/kevin/datasets -cf - af60v8f_base/lerobot/ds af60v8f_ds_dagger1/lerobot/ds af60v8f_ds_dagger2/lerobot/ds T20maxf_base/lerobot/ds T20maxf_ds_dagger1/lerobot/ds T20maxf_ds_dagger2/lerobot/ds |
    "${SSH[@]}" "$DEST" 'mkdir -p /workspace/kevin/datasets; tar --no-same-owner -C /workspace/kevin/datasets -xf - && touch /workspace/logs/datasets.ready'
) & datapid=$!
if [ -f ~/.cache/huggingface/token ]; then
  "${SSH[@]}" "$DEST" 'umask 077; cat > /workspace/hf/token' < ~/.cache/huggingface/token
fi
rc=0
for pid in "$envpid" "$modelpid" "$datapid"; do wait "$pid" || rc=1; done
test "$rc" = 0
"${SSH[@]}" "$DEST" 'touch /workspace/logs/transfer.ready'
