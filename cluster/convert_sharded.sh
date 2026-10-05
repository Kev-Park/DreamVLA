#!/bin/bash
# Sharded drop-in for convert_isaac_hdf5_to_lerobot.py: same arguments, plus --shards N (default 8).
#
#   cluster/convert_sharded.sh --hdf5-root DIR --recursive --output-path OUT --dataset-name NAME \
#       --task-prompt "..." --fps 50 [--shards 8] [any other converter flag]
#
# 1. lists the inputs exactly as the converter does (sorted(root.glob('**/*.hdf5')) with --recursive),
# 2. splits that list into N CONTIGUOUS slices (symlinks named <global position>__<name>, so each
#    shard's own sort keeps the global order),
# 3. runs N converters in parallel, one per slice,
# 4. merges the shards in order with merge_lerobot.py into OUT/NAME -- identical to what one
#    converter over the whole list writes (verified byte for byte by verify_merge.py).
# stats.json / relative_stats.json are left for GR00T to generate at fine-tune time, as with the
# single-process converter.
set -u
here=$(cd "$(dirname "$0")" && pwd)
WBC=${GR00T_WHOLEBODYCONTROL_DIR:-$HOME/kevin/GR00T-WholeBodyControl}
VENV=$WBC/.venv_data_collection
SHARDS=8 ROOT="" OUT="" NAME="" RECURSIVE=0 PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --shards) SHARDS=$2; shift 2;;
    --hdf5-root) ROOT=$2; shift 2;;
    --output-path) OUT=$2; shift 2;;
    --dataset-name) NAME=$2; shift 2;;
    --recursive) RECURSIVE=1; shift;;
    *) PASS+=("$1"); shift;;
  esac
done
[ -n "$ROOT" ] && [ -n "$OUT" ] || { echo "usage: $0 --hdf5-root DIR --output-path OUT [--dataset-name NAME] [--shards N] [converter flags]" >&2; exit 2; }
DEST=$OUT; [ -n "$NAME" ] && [[ "$OUT" != *"$NAME" ]] && DEST=$OUT/$NAME
[ -e "$DEST" ] && { echo "[sharded] $DEST exists; refusing to overwrite" >&2; exit 2; }
source "$VENV/bin/activate"
TMP=$(mktemp -d "${DEST%/}.shards.XXXX")
trap 'rm -rf "$TMP"' EXIT

# --- 1+2. list exactly like the converter, split contiguously ---------------------------------
python - "$ROOT" "$RECURSIVE" "$SHARDS" "$TMP" << 'PY'
import os, sys
from pathlib import Path
root, rec, n, tmp = Path(sys.argv[1]), sys.argv[2] == "1", int(sys.argv[3]), Path(sys.argv[4])
files = sorted(root.glob("**/*.hdf5" if rec else "*.hdf5"))            # == converter's _list_rollouts
if not files:
    sys.exit(f"no .hdf5 under {root}")
n = max(1, min(n, len(files)))
bounds = [round(i * len(files) / n) for i in range(n + 1)]
for k in range(n):
    d = tmp / f"in_{k:02d}"; d.mkdir()
    for g in range(bounds[k], bounds[k + 1]):
        os.symlink(files[g].resolve(), d / f"{g:06d}__{files[g].name}")
print(f"[sharded] {len(files)} rollouts -> {n} contiguous shards of {bounds[1] - bounds[0]}..{max(b - a for a, b in zip(bounds, bounds[1:]))}")
PY
[ $? -eq 0 ] || exit 1

# --- 3. N converters in parallel -----------------------------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
pids=() shard_dirs=()
for d in "$TMP"/in_*; do
  k=${d##*_}
  ( cd "$WBC/gear_sonic/scripts" && python convert_isaac_hdf5_to_lerobot.py --hdf5-root "$d" \
      --output-path "$TMP/out" --dataset-name "shard_$k" "${PASS[@]}" > "$TMP/log_$k.txt" 2>&1 ) &
  pids+=($!); shard_dirs+=("$TMP/out/shard_$k")
done
fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "[sharded] shard $i FAILED:"; tail -5 "$TMP/log_$(printf %02d $i).txt"; fail=1; }
done
[ $fail -eq 0 ] || exit 1
echo "[sharded] $(grep -h '^\[done\] wrote' "$TMP"/log_*.txt | awk '{s+=$3} END {print s}') frames converted in ${#pids[@]} shards"

# --- 4. merge in input order ---------------------------------------------------------------------
python "$here/merge_lerobot.py" --shards "${shard_dirs[@]}" --out "$DEST" || exit 1
f=$(python -c "import json;d=json.load(open('$DEST/meta/info.json'));print(d['total_frames'], d['total_episodes'])")
echo "[done] wrote ${f% *} frames across ${f#* } episode(s) -> $DEST   (sharded x${#pids[@]})"
