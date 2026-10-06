#!/bin/bash
# refine_batch.sh <jobs_file> "<gpu list>" <workers_per_gpu>
#
# Runs Adapter B (holosoma_to_pkl.py) over many clips on several GPUs at once. The refine recipe is
# whatever HS_* variables the caller exports (nothing is set here), so this is a drop-in for a
# hand-rolled loop over holosoma_to_pkl.py.
#
#   jobs_file: one "<retarget npz> <out base>" per line  ->  writes <out base>.pkl, log in
#              <dir of out base>/_logs/<basename>.log (the layout gen_dataset.sh / filter checks use)
#
# Safe to run several batches over overlapping job lists -- on one machine or on clusters sharing
# nothing: a clip is skipped if its .pkl already exists, and on a shared filesystem each clip is
# claimed atomically (mkdir <out base>.claim) so two workers never refine the same clip. A claim
# whose .pkl never appears (crashed worker) is cleared by re-running with REFINE_RECLAIM=1.
set -u
JOBS=${1:?usage: refine_batch.sh <jobs_file> "<gpu list>" <workers_per_gpu>}; GPUS=${2:-0}; NPG=${3:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "${REFINE_RECLAIM:-0}" = 1 ]; then
  while read -r npz out; do [ -n "$out" ] && [ ! -f "$out.pkl" ] && rmdir "$out.claim" 2>/dev/null; done < "$JOBS"
fi
worker() {   # gpu, slot, n_slots
  local gpu=$1 slot=$2 n=$3 i=0
  while read -r npz out; do
    [ -z "$out" ] && continue
    if [ $((i % n)) -eq "$slot" ] && [ ! -f "$out.pkl" ] && mkdir "$out.claim" 2>/dev/null; then
      mkdir -p "$(dirname "$out")/_logs"
      CUDA_VISIBLE_DEVICES=$gpu python "$HERE/holosoma_to_pkl.py" "$npz" "$out" > "$(dirname "$out")/_logs/$(basename "$out" | sed 's/^pick_//').log" 2>&1
      [ -f "$out.pkl" ] && rmdir "$out.claim" 2>/dev/null
    fi
    i=$((i + 1))
  done < "$JOBS"
}
slots=0; for g in $GPUS; do slots=$((slots + NPG)); done
s=0; t0=$(date +%s)
for g in $GPUS; do for _ in $(seq 1 "$NPG"); do worker "$g" "$s" "$slots" & s=$((s + 1)); done; done
wait
done_n=0; total=0
while read -r npz out; do [ -z "$out" ] && continue; total=$((total + 1)); [ -f "$out.pkl" ] && done_n=$((done_n + 1)); done < "$JOBS"
echo "[refine_batch] $done_n/$total clips have a .pkl ($slots workers on GPUs: $GPUS, $(( $(date +%s) - t0 ))s)"
