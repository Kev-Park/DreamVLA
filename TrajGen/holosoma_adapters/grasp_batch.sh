#!/bin/bash
# hsretargeting env location differs per box (bluesclues: ~/.holosoma_deps; biped: ~/kevin/.holosoma_deps).
for A in ~/.holosoma_deps/miniconda3/bin/activate ~/kevin/.holosoma_deps/miniconda3/bin/activate; do
  [ -f "$A" ] && { source "$A" hsretargeting; break; }
done
# Adapters live beside THIS script (repo TrajGen/holosoma_adapters), not a hardcoded ~/kevin path.
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
NPZ_DIR=${HS_NPZ_DIR:-$(pooled hs_pick_out)}; mkdir -p "$NPZ_DIR"
R=~/kevin/eval_videos/graspval; mkdir -p $R; CSV=$R/grasp_results.csv
echo "motion_id,status,held,grab,moved_max,lift,end_dist" > $CSV
for id in "$@"; do
  python "$SCRIPT_DIR/export_to_holosoma.py" $id > /dev/null 2>&1 || { echo "$id,adapterA_fail,,,,," >> $CSV; continue; }
  OUT=$NPZ_DIR/pick_${id}_original.npz; rm -f $OUT
  cd $HS
  timeout 300 python examples/robot_retarget.py --task-type object_interaction --robot g1 --data-format smplx --task-name pick_$id --data-path "$HS_INPUT" --save-dir "$NPZ_DIR" --task-config.object-name mustard > /tmp/hs_grasp_$id.log 2>&1
  if [ ! -f "$OUT" ]; then st=holosoma_fail; grep -qi infeasible /tmp/hs_grasp_$id.log && st=infeasible; echo "$id,$st,,,,," >> $CSV; continue; fi
  python "$SCRIPT_DIR/grasp_check.py" "$OUT" "$HS/models/g1/g1_29dof_w_mustard.xml" $id >> $CSV 2>/dev/null || echo "$id,check_fail,,,,," >> $CSV
done
echo "=== BATCH DONE $(date -u +%H:%MZ) ==="; column -t -s, $CSV
