#!/bin/bash
# gen_dataset.sh <gpu> <out_dir> <id...>
# One worker: for each motion id -> Adapter A (hsretargeting) -> holosoma retarget (hsretargeting,
# CPU) -> Adapter B with AL refine (dreamcontrol_51, GPU <gpu>) -> <out_dir>/pick_<id>.pkl.
# Launch several of these pinned to different GPUs to parallelize the full filtered dataset.
gpu=$1; out=$2; shift 2
# Optional overrides (default = holosoma defaults, byte-identical to the original pipeline):
#   HS_FOOT_STICK_TOL : --retargeter.foot-sticking-tolerance (default 1e-3). LOWER = stricter
#                       per-frame XY window; foot sticking is relative to the previous frame,
#                       so a tighter window slows accumulated drift over a clip.
#   HS_NPZ_DIR        : holosoma --save-dir (default ~/kevin/hs_pick_out). Point elsewhere to
#                       avoid clobbering the retarget output an existing dataset was built from.
#   HS_COM_MODE       : "" (off, default) | "full" | "rest". Enables the CoM static-stability
#                       barrier in the holosoma SQP (discrete-time CBF). "rest" restricts
#                       enforcement to the settled phase and needs HS_REST_MAP.
#   HS_REST_MAP       : file of "<id> <rest_start_frame>" lines, used by HS_COM_MODE=rest.
#   HS_COM_GAMMA      : CBF rate (default 0.5).  HS_COM_MARGIN: polygon inset in m (default 0.02).
#   HS_REFINE_ARM     : 1 (default) runs the AL right-arm refine in Adapter B; 0 skips it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HS=~/kevin/holosoma/src/holosoma_retargeting/holosoma_retargeting
HS_ACT=""
for A in ~/.holosoma_deps/miniconda3/bin/activate ~/kevin/.holosoma_deps/miniconda3/bin/activate; do
  [ -f "$A" ] && { HS_ACT="$A"; break; }
done
mkdir -p "$out"
for id in "$@"; do
  ( source "$HS_ACT" hsretargeting; python "$SCRIPT_DIR/export_to_holosoma.py" "$id" ) > /tmp/_gd_A_$id.log 2>&1 || { echo "$id ADAPTERA_FAIL"; continue; }
  NPZ_DIR=${HS_NPZ_DIR:-~/kevin/hs_pick_out}
  NPZ_DIR=$(eval echo "$NPZ_DIR"); mkdir -p "$NPZ_DIR"
  OUT=$NPZ_DIR/pick_${id}_original.npz; rm -f "$OUT"
  # CoM static-stability barrier flags (empty unless HS_COM_MODE is set)
  COM_ARGS=""
  if [ "$HS_COM_MODE" = "full" ]; then
    COM_ARGS="--retargeter.com-stability.enable --retargeter.com-stability.gamma ${HS_COM_GAMMA:-0.5} --retargeter.com-stability.margin ${HS_COM_MARGIN:-0.02}"
  elif [ "$HS_COM_MODE" = "rest" ]; then
    RS=$(awk -v i="$id" '$1==i{print $2}' "${HS_REST_MAP:-/dev/null}" 2>/dev/null)
    if [ -n "$RS" ]; then
      COM_ARGS="--retargeter.com-stability.enable --retargeter.com-stability.rest-only --retargeter.com-stability.rest-start-frame $RS --retargeter.com-stability.gamma ${HS_COM_GAMMA:-0.5} --retargeter.com-stability.margin ${HS_COM_MARGIN:-0.02}"
    else
      echo "$id NO_REST_START (skipping CoM barrier)"
    fi
  fi
  ( source "$HS_ACT" hsretargeting; cd "$HS"; timeout 400 python examples/robot_retarget.py \
      --task-type object_interaction --robot g1 --data-format smplx --task-name "pick_$id" \
      --data-path ~/kevin/hs_input --save-dir "$NPZ_DIR" --task-config.object-name mustard       ${HS_FOOT_STICK_TOL:+--retargeter.foot-sticking-tolerance $HS_FOOT_STICK_TOL} $COM_ARGS \
    ) > /tmp/_gd_HS_$id.log 2>&1
  [ -f "$OUT" ] || { echo "$id HOLOSOMA_FAIL"; continue; }
  ( source ~/miniconda3/etc/profile.d/conda.sh; conda activate dreamcontrol_51
    export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="$gpu"
    HS_REFINE_MODE=al HS_REFINE_ARM=${HS_REFINE_ARM:-1} HS_PKL_DOF=29 python "$SCRIPT_DIR/holosoma_to_pkl.py" "$OUT" "$out/pick_$id" \
    ) > /tmp/_gd_B_$id.log 2>&1
  [ -f "$out/pick_$id.pkl" ] && echo "$id OK" || echo "$id ADAPTERB_FAIL"
done
echo "[worker gpu$gpu] DONE $(date -u +%H:%MZ)"
