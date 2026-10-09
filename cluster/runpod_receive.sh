#!/bin/bash
# Run on bluesclues. Verify incoming tensors before replacing a competing run.
set -euo pipefail
HOST=${1:?}; PORT=${2:?}; RUN=${3:?}; STEPS=${4:?}; LINE=${5:?}
[[ "$RUN" =~ ^[A-Za-z0-9_]+$ && "$STEPS" =~ ^[0-9]+$ && "$LINE" =~ ^[A-Za-z0-9_]+$ ]]
ROOT=$HOME/kevin
mkdir -p "$ROOT/runpod_transport/received"
FAILED="$ROOT/runpod_transport/received/$RUN.failed"
trap 'rc=$?; printf "exit=%s line=%s time=%s\n" "$rc" "$LINENO" "$(date -u +%FT%TZ)" > "$FAILED"; exit "$rc"' ERR
rm -f "$FAILED"
SSH=(ssh -c aes128-gcm@openssh.com -i "$ROOT/runpod_transport/key" -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT")
PACK=/workspace/packs/$RUN
"${SSH[@]}" "root@$HOST" "mkdir -p /workspace/packs; if ! test -s $PACK/manifest.json || ! test -s $PACK/trained.safetensors; then HF_HOME=/workspace/hf /root/kevin/Isaac-GR00T/.venv/bin/python /workspace/kevin/DreamVLA_runtime/cluster/ckpt_transfer.py pack /root/kevin/checkpoints/$RUN/checkpoint-$STEPS $PACK; fi"
mkdir -p "$ROOT/runpod_transport/packs/$RUN" "$ROOT/runpod_transport/received"
printf -v RSYNC_SSH '%q ' "${SSH[@]}"
rsync -a --partial --info=progress2 --exclude='/optimizer.pt' --exclude='/scheduler.pt' --exclude='/rng_state*.pth' -e "$RSYNC_SSH" "root@$HOST:$PACK/" "$ROOT/runpod_transport/packs/$RUN/"
INCOMING="$ROOT/checkpoints/$RUN/runpod_checkpoint-$STEPS"
"$ROOT/Isaac-GR00T/.venv/bin/python" "$ROOT/wt/dagger-watch-sparse/cluster/ckpt_transfer.py" unpack \
  "$ROOT/runpod_transport/packs/$RUN" "$ROOT/checkpoints/af60v7f_run01/checkpoint-10000" "$INCOMING"
# These exact experiment names belong to this orchestrator. Only cancel after
# the replacement has passed all tensor checksums.
pkill -TERM -f "[e]xperiment-name $RUN --max-steps" || true
TARGET="$ROOT/checkpoints/$RUN/checkpoint-$STEPS"
if [ -e "$TARGET" ]; then mv "$TARGET" "${TARGET}.superseded.$(date +%s)"; fi
mv "$INCOMING" "$TARGET"
mkdir -p "$ROOT/dline/$LINE/ftq"
rm -f "$ROOT/dline/$LINE/ftq/$RUN.failed"
echo "runpod $(date -u +%H:%M)" > "$ROOT/dline/$LINE/ftq/$RUN.ready"
echo "$STEPS" > "$ROOT/runpod_transport/received/$RUN"
