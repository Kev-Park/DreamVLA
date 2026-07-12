#!/bin/bash
# gen_dataset.sh <gpu> <out_dir> <id...>
# One worker: for each motion id -> Adapter A (hsretargeting) -> holosoma retarget (hsretargeting,
# CPU) -> Adapter B with AL refine (dreamcontrol_51, GPU <gpu>) -> <out_dir>/pick_<id>.pkl.
# Launch several of these pinned to different GPUs to parallelize the full filtered dataset.
gpu=$1; out=$2; shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HS=~/kevin/holosoma/src/holosoma_retargeting/holosoma_retargeting
HS_ACT=""
for A in ~/.holosoma_deps/miniconda3/bin/activate ~/kevin/.holosoma_deps/miniconda3/bin/activate; do
  [ -f "$A" ] && { HS_ACT="$A"; break; }
done
mkdir -p "$out"
for id in "$@"; do
  ( source "$HS_ACT" hsretargeting; python "$SCRIPT_DIR/export_to_holosoma.py" "$id" ) > /tmp/_gd_A_$id.log 2>&1 || { echo "$id ADAPTERA_FAIL"; continue; }
  OUT=~/kevin/hs_pick_out/pick_${id}_original.npz; rm -f "$OUT"
  ( source "$HS_ACT" hsretargeting; cd "$HS"; timeout 400 python examples/robot_retarget.py \
      --task-type object_interaction --robot g1 --data-format smplx --task-name "pick_$id" \
      --data-path ~/kevin/hs_input --save-dir ~/kevin/hs_pick_out --task-config.object-name mustard \
    ) > /tmp/_gd_HS_$id.log 2>&1
  [ -f "$OUT" ] || { echo "$id HOLOSOMA_FAIL"; continue; }
  ( source ~/miniconda3/etc/profile.d/conda.sh; conda activate dreamcontrol_51
    export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="$gpu"
    HS_REFINE_MODE=al HS_REFINE_ARM=1 HS_PKL_DOF=29 python "$SCRIPT_DIR/holosoma_to_pkl.py" "$OUT" "$out/pick_$id" \
    ) > /tmp/_gd_B_$id.log 2>&1
  [ -f "$out/pick_$id.pkl" ] && echo "$id OK" || echo "$id ADAPTERB_FAIL"
done
echo "[worker gpu$gpu] DONE $(date -u +%H:%MZ)"
