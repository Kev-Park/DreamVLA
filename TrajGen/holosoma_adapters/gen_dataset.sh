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
#   HS_INPUT_DIR      : Adapter A output / holosoma --data-path (default: pooled ~/kevin/hs_input).
#   HOLOSOMA_DIR      : holosoma checkout to retarget with. Defaults to a holosoma worktree
#                       beside THIS checkout if one exists, else the shared ~/kevin/holosoma.
#                       Set it to run a variant retargeter without moving anything.
#   HS_NPZ_DIR        : holosoma --save-dir (default: pooled ~/kevin/hs_pick_out). The pool is
#                       shared ON PURPOSE so branches reuse one dataset -- but a run with a
#                       NON-DEFAULT retargeter config MUST point this elsewhere, or it
#                       overwrites the pooled pick_<id>_original.npz. Point elsewhere to
#                       avoid clobbering the retarget output an existing dataset was built from.
#   HS_COM_MODE       : "" (off, default) | "full" | "rest". Enables the CoM static-stability
#                       barrier in the holosoma SQP (discrete-time CBF). "rest" restricts
#                       enforcement to the settled phase and needs HS_REST_MAP.
#   HS_REST_MAP       : file of "<id> <rest_start_frame>" lines, used by HS_COM_MODE=rest.
#   HS_COM_GAMMA      : CBF rate (default 0.5).  HS_COM_MARGIN: polygon inset in m (default 0.02).
#   HS_COM_SLACK      : L1 penalty on the per-edge relaxation slack (holosoma default 50). The
#                       barrier is a hard constraint RELAXED by a penalised slack, so this is the
#                       hard-vs-objective knob: <=0 = strictly hard; ~5-20 = stability negotiates
#                       with mesh tracking (laplacian_weights=10) as a soft objective; 1e4 = hard.
#   HS_COM_RAMP       : frames over which the polygon margin fades in before the rest start
#                       (holosoma default 10 = 0.5 s @20fps). The fade is a C2 smootherstep and is
#                       clamped to the rest start; longer = gentler arrival of the constraint.
#   HS_REST_MAP       : with HS_COM_MODE=rest, OPTIONAL. Unset => holosoma derives the rest start
#                       from the contact schedule (preferred).
#   HS_REFINE_ARM     : 1 (default) runs the AL right-arm refine in Adapter B; 0 skips it.
#   HS_NO_FOOT_STICK  : 1 disables the foot-sticking constraint entirely
#                       (--retargeter.no-activate-foot-sticking). Ablation: foot sticking removes
#                       ~93% of the source motion's foot slip, but slip-removal correlates -0.47
#                       with ZMP feasibility, so pinning the feet while the body still tracks the
#                       human appears to cost dynamic feasibility. This isolates that trade.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # .../TrajGen/holosoma_adapters -> repo root
SHARED_ROOT=~/kevin
# Mirrors Training/source/isaaclab_tasks/isaaclab_tasks/utils/repo_paths.py.
#   sibling = CODE, worktree-local first: a worktree with its own holosoma worktree beside it uses
#             THAT one, so two branches can develop different retargeters concurrently; a worktree
#             without a paired clone falls through to the shared ~/kevin copy.
#   pooled  = DATA, shared first: retargeted datasets are expensive and branches normally want the
#             same best/most-recent one.
# Either is overridden by an env var named after the directory: HOLOSOMA_DIR, HS_INPUT_DIR, ...
_pick() { local n=$1; shift; local ov
  ov=$(eval echo "\$$(echo "$n" | tr 'a-z-' 'A-Z_')_DIR")
  [ -n "$ov" ] && { eval echo "$ov"; return; }
  for r in "$@"; do [ -e "$r/$n" ] && { echo "$r/$n"; return; }; done
  echo "$1/$n"; }
sibling() { _pick "$1" "$(dirname "$REPO_ROOT")" "$SHARED_ROOT" ~; }
pooled()  { _pick "$1" "$SHARED_ROOT" "$(dirname "$REPO_ROOT")" ~; }
HS=$(sibling holosoma)/src/holosoma_retargeting/holosoma_retargeting
HS_INPUT=$(pooled hs_input)
HS_ACT=""
for A in ~/.holosoma_deps/miniconda3/bin/activate ~/kevin/.holosoma_deps/miniconda3/bin/activate; do
  [ -f "$A" ] && { HS_ACT="$A"; break; }
done
mkdir -p "$out"
LOG_DIR="$out/_logs"; mkdir -p "$LOG_DIR"
for id in "$@"; do
  ( source "$HS_ACT" hsretargeting; python "$SCRIPT_DIR/export_to_holosoma.py" "$id" ) > "$LOG_DIR/A_$id.log" 2>&1 || { echo "$id ADAPTERA_FAIL"; continue; }
  NPZ_DIR=${HS_NPZ_DIR:-$(pooled hs_pick_out)}; mkdir -p "$NPZ_DIR"
  OUT=$NPZ_DIR/pick_${id}_original.npz; rm -f "$OUT"
  # CoM static-stability barrier flags (empty unless HS_COM_MODE is set)
  COM_ARGS=""
  COM_COMMON="--retargeter.com-stability.gamma ${HS_COM_GAMMA:-0.5} --retargeter.com-stability.margin ${HS_COM_MARGIN:-0.02}${HS_COM_SLACK:+ --retargeter.com-stability.slack-penalty $HS_COM_SLACK}${HS_COM_RAMP:+ --retargeter.com-stability.ramp-frames $HS_COM_RAMP}"
  if [ "$HS_COM_MODE" = "full" ]; then
    COM_ARGS="--retargeter.com-stability.enable $COM_COMMON"
  elif [ "$HS_COM_MODE" = "rest" ]; then
    if [ -z "$HS_REST_MAP" ]; then
      # No explicit map: let holosoma DERIVE the rest start from the contact schedule
      # (first frame of the final contiguous double-support run). Contact-based, so it
      # admits a settled pose that still carries momentum -- which a velocity threshold
      # wrongly excludes. Preferred over an externally computed velocity-based map.
      COM_ARGS="--retargeter.com-stability.enable --retargeter.com-stability.rest-only $COM_COMMON"
    else
      RS=$(awk -v i="$id" '$1==i{print $2}' "$HS_REST_MAP" 2>/dev/null)
      if [ -n "$RS" ]; then
        COM_ARGS="--retargeter.com-stability.enable --retargeter.com-stability.rest-only --retargeter.com-stability.rest-start-frame $RS $COM_COMMON"
      else
        echo "$id NO_REST_START (skipping CoM barrier)"
      fi
    fi
  fi
  ( source "$HS_ACT" hsretargeting; cd "$HS"; timeout 400 python examples/robot_retarget.py \
      --task-type object_interaction --robot g1 --data-format smplx --task-name "pick_$id" \
      --data-path "$HS_INPUT" --save-dir "$NPZ_DIR" --task-config.object-name mustard       ${HS_FOOT_STICK_TOL:+--retargeter.foot-sticking-tolerance $HS_FOOT_STICK_TOL} $COM_ARGS \
      ${HS_NO_FOOT_STICK:+--retargeter.no-activate-foot-sticking} \
    ) > "$LOG_DIR/HS_$id.log" 2>&1
  [ -f "$OUT" ] || { echo "$id HOLOSOMA_FAIL"; continue; }
  ( source ~/miniconda3/etc/profile.d/conda.sh; conda activate dreamcontrol_51
    export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="$gpu"
    HS_REFINE_MODE=al HS_REFINE_ARM=${HS_REFINE_ARM:-1} HS_PKL_DOF=29 python "$SCRIPT_DIR/holosoma_to_pkl.py" "$OUT" "$out/pick_$id" \
    ) > "$LOG_DIR/B_$id.log" 2>&1
  [ -f "$out/pick_$id.pkl" ] && echo "$id OK" || echo "$id ADAPTERB_FAIL"
done
echo "[worker gpu$gpu] DONE $(date -u +%H:%MZ)"
