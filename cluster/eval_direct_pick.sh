#!/usr/bin/env bash
# User-requested direct-actor ablation, same held-out pool/seeds/physics/four-rule score.
set -euo pipefail
MODE=${1:-snap}
[[ "$MODE" == raw || "$MODE" == snap ]] || exit 2
CODE=$(cd "$(dirname "$0")/.." && pwd)
RUN=pick_expert_direct_$MODE
W=$HOME/kevin/dline/$RUN
OUT=$HOME/kevin/eval_videos
mkdir -p "$W" "$OUT"
[[ ! -f "$W/ev_$RUN.done" ]] || exit 0
[[ ! -e "$W/started" ]] || { echo 'Existing attempt: inspect before restarting'; exit 1; }
touch "$W/started"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "ABORT exit $rc at $(date -u)" >> "$W/status"; touch "$W/abort"; fi' EXIT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamcontrol_51
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HS_REWORK=1 HS_REWORK_OBJ_PC=1 HS_REWORK_OBJ_GATE=1 HS_REWORK_CONTACT_LEAD=2 HS_EVAL_NO_EE_TERM=1
RSL=$CODE/Training/scripts/reinforcement_learning/rsl_rl
export PYTHONPATH=$RSL${PYTHONPATH:+:$PYTHONPATH}
cp ~/kevin/dline/af60v8f57/box.json "$W/box.json"
python - "$W" <<'PY'
import pathlib,sys
w=pathlib.Path(sys.argv[1])
for k in range(4): (w/f'_ids_shard{k}.txt').write_text(' '.join(map(str,range(k,100,4))))
PY
cd "$CODE/Training"
echo "START $(date -u) code=$(git rev-parse HEAD) token=$MODE" >> "$W/status"
for wave in 0 1; do
  mapfile -t GPUS < <(python "$CODE/cluster/gpu_slots.py" reserve --owner $$ --count 2 --mb 10000 | tr ' ' '\n' | sed '/^$/d')
  [[ ${#GPUS[@]} -eq 2 ]] || { echo 'Insufficient free GPU slots' >&2; exit 1; }
  PIDS=()
  for offset in 0 1; do
    k=$((wave*2+offset)); gpu=${GPUS[$offset]}
    echo "shard=$k gpu=$gpu start $(date -u)" >> "$W/status"
    CUDA_VISIBLE_DEVICES=$gpu timeout 45m python "$RSL/eval_vla_sonic.py" --headless \
      --sonic-pt "$HOME/kevin/sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000" \
      --expert-checkpoint "$HOME/kevin/residuals/af60v8c_model_8998.pt" --expert-token-mode "$MODE" \
      --ref-motions-path Holosoma_Pick_29_latband60max_eval --waist-dof 29 \
      --motions-from "$W/_ids_shard$k.txt" --num-episodes 25 --seed $((100+k)) \
      --traj-dump "$OUT/${RUN}_b$k" > "$W/ev_${RUN}_$k.log" 2>&1 &
    PIDS+=("$!")
  done
  failed=0
  for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
  for gpu in "${GPUS[@]}"; do python "$CODE/cluster/gpu_slots.py" release --owner $$ --gpu "$gpu"; done
  [[ $failed -eq 0 ]] || exit 1
done
python - "$W" "$OUT" "$RUN" "$MODE" <<'PY'
import json,pathlib,sys,numpy as np
from vla_sonic.grasp_success import score
w,out,run,mode=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3],sys.argv[4]
files=sorted(out.glob(run+'_b*_traj.npz'))
assert len(files)==100, f'Incomplete evaluation: {len(files)}/100'
b=json.loads((w/'box.json').read_text()); rows=[]
for f in files:
    z=np.load(f)
    assert len(z['obj_pos']) and all(np.isfinite(z[k]).all() for k in ['obj_pos','obj_quat','root_pos','root_quat','wrist'])
    result=score(z['obj_pos'],z['obj_quat'],z['root_pos'],z['root_quat'],z['wrist'],box_lo=np.array(b['lo']),box_hi=np.array(b['hi']))
    rows.append(dict(file=f.name,success=result['success'],**{k:result[k] for k in ['fallen_height','fallen_horizontal','in_box_at_end','slipping']}))
passed=sum(x['success'] for x in rows)
result=dict(run=run,token_mode=mode,checkpoint='af60v8c_model_8998.pt',pool='Holosoma_Pick_29_latband60max_eval',
            successes=passed,episodes=100,success_rate=passed/100,criterion='existing calibrated four-rule',results=rows)
(out/(run+'_success.json')).write_text(json.dumps(result,indent=2))
with (w/'status').open('a') as f:f.write(f'4-rule: {passed}/100 = {passed}%\n')
print(f'4-rule: {passed}/100 = {passed}%',flush=True)
PY
python "$CODE/cluster/mpjpe_report.py" "$OUT/${RUN}_b*_traj.npz" --output "$OUT/${RUN}_mpjpe.json" >> "$W/status" 2>&1
touch "$W/ev_$RUN.done"
echo "DONE $(date -u)" >> "$W/status"
