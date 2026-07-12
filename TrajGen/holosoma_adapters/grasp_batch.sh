#!/bin/bash
# hsretargeting env location differs per box (bluesclues: ~/.holosoma_deps; biped: ~/kevin/.holosoma_deps).
for A in ~/.holosoma_deps/miniconda3/bin/activate ~/kevin/.holosoma_deps/miniconda3/bin/activate; do
  [ -f "$A" ] && { source "$A" hsretargeting; break; }
done
# Adapters live beside THIS script (repo TrajGen/holosoma_adapters), not a hardcoded ~/kevin path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HS=~/kevin/holosoma/src/holosoma_retargeting/holosoma_retargeting
R=~/kevin/eval_videos/graspval; mkdir -p $R; CSV=$R/grasp_results.csv
echo "motion_id,status,held,grab,moved_max,lift,end_dist" > $CSV
for id in "$@"; do
  python "$SCRIPT_DIR/export_to_holosoma.py" $id > /dev/null 2>&1 || { echo "$id,adapterA_fail,,,,," >> $CSV; continue; }
  OUT=~/kevin/hs_pick_out/pick_${id}_original.npz; rm -f $OUT
  cd $HS
  timeout 300 python examples/robot_retarget.py --task-type object_interaction --robot g1 --data-format smplx --task-name pick_$id --data-path ~/kevin/hs_input --save-dir ~/kevin/hs_pick_out --task-config.object-name mustard > /tmp/hs_grasp_$id.log 2>&1
  if [ ! -f "$OUT" ]; then st=holosoma_fail; grep -qi infeasible /tmp/hs_grasp_$id.log && st=infeasible; echo "$id,$st,,,,," >> $CSV; continue; fi
  python "$SCRIPT_DIR/grasp_check.py" "$OUT" "$HS/models/g1/g1_29dof_w_mustard.xml" $id >> $CSV 2>/dev/null || echo "$id,check_fail,,,,," >> $CSV
done
echo "=== BATCH DONE $(date -u +%H:%MZ) ==="; column -t -s, $CSV
