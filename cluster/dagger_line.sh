#!/bin/bash
# Generic GR00T distillation line: base fine-tune + NR DAgger rounds (no early stop), optional label filter.
#   TAG      run prefix (checkpoints <TAG>_run01, <TAG>_dagger<r>_run01; datasets ~/kevin/datasets/<TAG>_*)
#   CK       residual expert checkpoint            REFS     reference set the expert/student run on
#   EVREFS   eval reference set                    EVN      eval episodes (4 shards; default 100)
#   EVIDS    "all" (ids 0..|EVREFS|-1) or a dir holding _ids_shard{0..3}.txt
#   BASE     "collect" (one pass of the expert over every REFS clip) or "keep:<file of hdf5 paths>"
#   BOX      "calibrate" (from the base demos) or a box json path
#   FILTER   none | med3 (3-frame median per token dim on action.motion_token before every fine-tune)
#   NR       DAgger rounds (default 5)     NSH  collection shards (default 8)     STEPS fine-tune steps (10000)
#   CODE     DreamVLA checkout the expert was trained in (default ~/kevin/DreamVLA; a worktree for other branches)
#   BRANCH   branch CODE fast-forwards to at start (default g1)    VENV  python venv to activate instead of conda dreamcontrol_51
#   ENVX     extra "K=V ..." env exported for collection/eval/render (the expert's training flags)
#   ROLL     collection rollout cap in steps (default 500)        XARGS extra args for collect/eval/play (e.g. --skip-start-frames 0)
set -u
if [ -n "${VENV:-}" ]; then source $VENV/bin/activate; else source ~/miniconda3/etc/profile.d/conda.sh; conda activate dreamcontrol_51; fi
[ -n "${ENVX:-}" ] && export ${ENVX}
CODE=${CODE:-$HOME/kevin/DreamVLA}; BRANCH=${BRANCH:-g1}; ROLL=${ROLL:-500}; XARGS=${XARGS:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HS_REWORK=1 HS_REWORK_OBJ_PC=1 HS_REWORK_OBJ_GATE=1 HS_REWORK_CONTACT_LEAD=2
: ${TAG:?} ${CK:?} ${REFS:?} ${EVREFS:?} ${BASE:?}; EVN=${EVN:-100}; EVIDS=${EVIDS:-all}; BOX=${BOX:-calibrate}
FILTER=${FILTER:-none}; NR=${NR:-5}; NSH=${NSH:-8}; STEPS=${STEPS:-10000}
# DAGGER_N>0: each DAgger round collects on a seeded random DAGGER_N of the reference motions (always including
# the first 8 demos by name, which the collector uses for its closed-hand pose) and keeps at most DAGGER_N episodes.
DAGGER_N=${DAGGER_N:-0}
# FT_PASSES>0: steps per fine-tune = FT_PASSES * dataset frames / batch 16, rounded up to 1000, clamped to
# [STEPS, FT_MAX_STEPS] (af60v7's 10k-step base = 2.3 passes over its 70k frames). FT_EXTERNAL=1: hand each
# fine-tune to an external runner (request file in $W/ftq, see cluster orchestration) and fall back to a local
# bluesclues fine-tune if nobody claims it within FT_CLAIM_WAIT minutes or the runner reports failure.
FT_PASSES=${FT_PASSES:-0}; FT_MAX_STEPS=${FT_MAX_STEPS:-30000}; FT_EXTERNAL=${FT_EXTERNAL:-0}; FT_CLAIM_WAIT=${FT_CLAIM_WAIT:-15}
W=$HOME/kevin/dline/$TAG; mkdir -p $W; ST=$W/status; touch $ST
log(){ echo "$(date -u +%H:%M) $*" >> $ST; }
die(){ log "ABORT: $*"; touch $W/abort; exit 1; }
WT=$CODE; RSL=$WT/Training/scripts/reinforcement_learning/rsl_rl; DS=$HOME/kevin/datasets; OUTV=$HOME/kevin/eval_videos
PT=$HOME/kevin/sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000
PROMPT="pick up the mustard bottle"
freegpus(){ local n=$1 got; for _ in $(seq 1 60); do
    got=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}' | sort -rn | head -$n | sort -n | tr '\n' ' ')
    [ $(echo $got | wc -w) -ge $n ] && break; sleep 30; done; echo $got; }
if [ "${SKIP_CODE_UPDATE:-0}" != 1 ]; then cd $WT && git fetch -q origin && git merge -q --ff-only origin/$BRANCH; fi; cd $WT; log "start $TAG: CODE=$CODE@$BRANCH ENVX=[${ENVX:-}] ROLL=$ROLL XARGS=[$XARGS] CK=$CK REFS=$REFS EVREFS=$EVREFS BASE=$BASE BOX=$BOX FILTER=$FILTER NR=$NR; checkout $(git log --oneline -1 | cut -c1-50)"
cd $WT/Training
NREF=$(ls $HOME/kevin/ref_motions/$REFS/pick_*.pkl | wc -l); NEV=$(ls $HOME/kevin/ref_motions/$EVREFS/pick_*.pkl | wc -l)
[ $NREF -gt 0 ] && [ $NEV -gt 0 ] || die "missing reference sets ($REFS: $NREF, $EVREFS: $NEV)"
if [ "$EVIDS" = all ]; then
  python -c "
n=$NEV
for k in range(4): open('$W/_ids_shard%d.txt' % k, 'w').write(' '.join(str(i) for i in range(k, n, 4)) + '\n')"
  EVIDS=$W
fi
COMMON="--headless --task Isaac-Motion-Tracking-Pick-Cam-HOI-v0 --waist-dof 29 --sonic-pt $PT --encoder-mode g1 \
  --residual-transform additive_free --residual-scale 0.1 --checkpoint-path $CK --ref-motions-path $REFS --rollout-length $ROLL --num-samples 100000"
ranges(){ python -c "
n, k = $NREF, ${1:-$NSH}
b = [round(i * n / k) for i in range(k + 1)]
print(' '.join(f'{b[i]},{b[i+1]}' for i in range(k)))"; }

freegpus_upto(){ local n=$1 m=$2 got; for _ in $(seq 1 120); do
    got=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}' | sort -rn | head -$n | sort -n | tr '
' ' ')
    [ $(echo $got | wc -w) -ge $m ] && break; sleep 30; done; echo $got; }
gpu_slots(){  # want [slot MB] -> GPU index per slot of free memory (one process per slot), fullest-first avoided
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | sort -t, -k2 -n     | awk -F', ' -v w=$1 -v sz=${2:-15500} '{k=int(($3-2000-$2)/sz); for(i=0;i<k && n<w;i++){print $1; n++}}' | tr '
' ' '; }
collect(){  # outdir seed [extra collector args...]
  # Shards go wherever there is GPU memory (~15 GB per collector, shared with the other line); up to NSH of them,
  # at least 4. The motion ranges follow the shard count; --skip-existing makes any re-split safe.
  local OUT=$1 S=$2; shift 2; mkdir -p $OUT; local G=() t=0
  while :; do G=($(gpu_slots $NSH)); [ ${#G[@]} -ge 4 ] && break; t=$((t + 1)); [ $t -ge 120 ] && die "fewer than 4 GPU slots for collection"; sleep 30; done
  local NS=${#G[@]}; log "    collecting on $NS shards (GPU slots: ${G[*]})"
  local i=0 r; local PIDS=()
  for r in $(ranges $NS); do
    local L=$OUT/_log_$i.txt
    ( CUDA_VISIBLE_DEVICES=${G[$i]} python ${COLLECT_SCRIPT:-$RSL/collect_sonic_adapter.py} $COMMON $XARGS --motion-range ${r%,*} ${r#*,} --seed $S \
        --skip-existing "$@" --output-directory $OUT > $L 2>&1; rc=$?; log "    shard $i exit $rc"; exit "$rc" ) &
    PIDS+=($!)
    until grep -q "\[INFO\] motion \|Traceback" $L 2>/dev/null; do sleep 15; done; sleep 15; i=$((i + 1))
  done
  local FAILED=0 P
  for P in "${PIDS[@]}"; do wait "$P" || FAILED=1; done
  return "$FAILED"
}
score_keep(){  # pattern keepfile tag -> writes passing hdf5 paths
  python - "$1" "$2" "$3" "$W/box.json" >> $ST 2>&1 << 'PY'
import sys, glob, os, json, numpy as np, h5py
sys.path.insert(0, os.path.expanduser("~/kevin/DreamVLA/Training/scripts/reinforcement_learning/rsl_rl"))
from vla_sonic.grasp_success import score
pat, keep, tag, boxf = sys.argv[1:5]; b = json.load(open(boxf)); lo, hi = np.array(b["lo"]), np.array(b["hi"])
fs = sorted(f for f in glob.glob(os.path.expanduser(pat), recursive=True) if "_passA" not in f); ok = []; why = {}
for f in fs:
    try:
        with h5py.File(f, "r", locking=False) as h:
            g = h["data/demo_0"]; r = score(g["obs/object_pos"][()], g["obs/object_quat"][()], g["obs/robot0_root_pos_w"][()], g["obs/robot0_root_quat_w"][()], g["teleop/right_wrist"][()], box_lo=lo, box_hi=hi)
    except Exception:
        why["unreadable"] = why.get("unreadable", 0) + 1; continue
    if r["success"]: ok.append(f)
    else:
        for k in ("fallen_height", "fallen_horizontal", "slipping"):
            if r[k]: why[k] = why.get(k, 0) + 1
        if not r["in_box_at_end"]: why["not_in_box"] = why.get("not_in_box", 0) + 1
open(keep, "w").write("\n".join(ok) + "\n")
print(f"  [{tag}] 4-rule: {len(ok)}/{len(fs)} pass; rejections {why}")
PY
}
build_ds(){  # name keepfiles... -> LeRobot dataset dir (echo)
  local NAME=$1; shift; local A=$DS/${TAG}_$NAME
  # A finished dataset is never rebuilt: a running fine-tune may be reading it (a chain restart once deleted
  # the base dataset out from under its fine-tune). built.ok is written only after conversion (+ filter).
  [ -f $A/built.ok ] && [ -f $A/lerobot/ds/meta/info.json ] && { log "  dataset ${TAG}_$NAME already built, reusing"; echo $A/lerobot/ds; return; }
  rm -rf $A; mkdir -p $A/hdf5; local i=0
  for kf in "$@"; do while read -r f; do [ -z "$f" ] && continue; ln -s "$f" "$A/hdf5/ep_$(printf '%05d' $i)__$(basename $f)"; i=$((i + 1)); done < $kf; done
  log "  dataset ${TAG}_$NAME: $i episodes"
  bash $HOME/kevin/DreamVLA/cluster/convert_sharded.sh --hdf5-root $A/hdf5 --recursive --output-path $A/lerobot --dataset-name ds \
      --task-prompt "$PROMPT" --fps 50 --shards 8 > $A/convert.log 2>&1 || die "conversion of $NAME failed"
  log "  $(grep -m1 '\[done\] wrote' $A/convert.log | cut -c1-70)"
  if [ "$FILTER" = med3 ]; then
    ( source $HOME/kevin/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
      python - $A/lerobot/ds >> $ST 2>&1 << 'PY'
import sys, glob, numpy as np, pyarrow as pa, pyarrow.parquet as pq, datasets
d = sys.argv[1]; ch = tot = 0
for f in sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True)):
    t = pq.read_table(f); i = t.schema.get_field_index("action.motion_token"); fld = t.schema.field(i)
    tok = np.stack(t.column(i).to_pylist()).astype(np.float32); med = tok.copy()
    med[1:-1] = np.median(np.stack([tok[:-2], tok[1:-1], tok[2:]]), axis=0); ch += int((med != tok).sum()); tot += tok.size
    datasets.Dataset(t.set_column(i, fld, pa.array(list(med), type=fld.type))).to_parquet(f)
print(f"  med3 label filter: changed {ch}/{tot} token entries ({100*ch/tot:.2f}%)")
PY
    )
  fi
  touch $A/built.ok
  echo $A/lerobot/ds
}
steps_of(){ cat $W/steps_$1 2>/dev/null || echo $STEPS; }   # steps the run was (or will be) trained for
finetune(){  # dataset run
  # Read a per-line policy at the boundary, never in the middle of a fine-tune.
  if [ -f "$W/finetune_policy.json" ]; then
    local POLICY
    POLICY=$(python -c "import json; p=json.load(open('$W/finetune_policy.json')); e=float(p['passes']); c=int(p['max_steps']); assert e>0 and c>=0; print(e,c)") || die "invalid fine-tune policy"
    read -r FT_PASSES FT_MAX_STEPS <<< "$POLICY"
  fi
  local FST=$STEPS
  if [ "$FT_PASSES" != 0 ] && [ ! -f $W/steps_$2 ]; then
    local NF=$(python -c "import json; print(json.load(open('$1/meta/info.json'))['total_frames'])")
    FST=$(python -c "import math; s=math.ceil($FT_PASSES*$NF/16/1000)*1000; c=$FT_MAX_STEPS; print(max($STEPS, min(c, s) if c>0 else s))")
    log "  $2: $NF frames x $FT_PASSES passes -> $FST steps"
  fi
  [ -f $W/steps_$2 ] || echo $FST > $W/steps_$2; FST=$(steps_of $2)
  local CK=$HOME/kevin/checkpoints/$2/checkpoint-$FST/model.safetensors.index.json
  [ -f $CK ] && { log "  $2 exists, skipping fine-tune"; return; }
  if [ "$FT_EXTERNAL" = 1 ]; then
    local Q=$W/ftq; mkdir -p $Q; [ -f $Q/$2.req ] || { echo "$1 $FST" > $Q/$2.req; log "  finetune $2 ($FST steps): requested external runner"; }
    local t=0; while [ ! -f $Q/$2.claimed ] && [ $t -lt $((FT_CLAIM_WAIT * 2)) ]; do sleep 30; t=$((t + 1)); done
    if [ -f $Q/$2.claimed ]; then
      log "  $2 claimed by: $(cat $Q/$2.claimed)"
      while [ ! -f $Q/$2.ready ] && [ ! -f $Q/$2.failed ]; do sleep 60; done
      [ -f $Q/$2.ready ] && [ -f $CK ] && { log "  finetune $2 ready ($(cat $Q/$2.ready))"; return; }
      log "  external runner failed for $2 ($(cat $Q/$2.failed 2>/dev/null)); falling back to bluesclues"
    else
      echo "chain-local $(date -u +%H:%M)" > $Q/$2.claimed; log "  $2 unclaimed after $FT_CLAIM_WAIT min; running on bluesclues"
    fi
  fi
  local STEPS=$FST
  local G=($(freegpus 1)); [ ${#G[@]} -ge 1 ] || die "no GPU for fine-tune"
  log "  finetune $2 ($STEPS steps, GPU ${G[0]})"
  ( cd $HOME/kevin/Isaac-GR00T && CUDA_VISIBLE_DEVICES=${G[0]} .venv/bin/python gr00t/experiment/launch_finetune.py \
      --base-model-path nvidia/GR00T-N1.7-3B --dataset-path $1 --embodiment-tag unitree_g1_sonic --num-gpus 1 --global-batch-size 16 \
      --dataloader-num-workers 8 --output-dir $HOME/kevin/checkpoints --experiment-name $2 --max-steps $STEPS --save-steps $STEPS \
      --save-total-limit 1 --save-only-model > $W/ft_$2.log 2>&1 )
  log "  finetune $2 exit $? $(grep -oE "train_loss.: [0-9.]+" $W/ft_$2.log | tail -1)"
  [ -f $HOME/kevin/checkpoints/$2/checkpoint-$STEPS/model.safetensors.index.json ] || die "no checkpoint for $2"
}
evaluate(){  # run
  local RUN=$1 VLA=$HOME/kevin/checkpoints/$1/checkpoint-$(steps_of $1) EPS=$((EVN / 4))
  [ -f $W/ev_$RUN.done ] && return
  # one GPU slot (~16 GB) per eval process; the 4 shards (fixed EVN/4 episodes, fixed seeds) never share a slot
  local G=() t=0; while :; do G=($(gpu_slots 4 16000)); [ ${#G[@]} -ge 4 ] && break; t=$((t + 1)); [ $t -ge 120 ] && die "no GPU room for eval"; sleep 30; done
  local NG=${#G[@]} j=0
  rm -f $OUTV/${RUN}_b*_traj.npz; log "  eval $RUN on $EVREFS: 4 x $EPS episodes (GPUs ${G[*]})"
  for k in 0 1 2 3; do
    HS_EVAL_NO_EE_TERM=1 CUDA_VISIBLE_DEVICES=${G[$((k % NG))]} python ${EVAL_SCRIPT:-$RSL/eval_vla_sonic.py} --headless $XARGS --sonic-pt $PT --ref-motions-path $EVREFS \
      --vla-checkpoint $VLA --waist-dof 29 --motions-from $EVIDS/_ids_shard$k.txt --num-episodes $EPS --seed $((100 + k)) \
      --traj-dump $OUTV/${RUN}_b$k > $W/ev_${RUN}_$k.log 2>&1 &
    sleep 20
  done; wait
  python - $RUN $W/box.json >> $ST 2>&1 << 'PY'
import sys, glob, os, json, math, numpy as np
sys.path.insert(0, os.path.expanduser("~/kevin/DreamVLA/Training/scripts/reinforcement_learning/rsl_rl"))
from vla_sonic.grasp_success import score
run = sys.argv[1]; b = json.load(open(sys.argv[2])); lo, hi = np.array(b["lo"]), np.array(b["hi"]); k = n = 0
for f in sorted(glob.glob(os.path.expanduser(f"~/kevin/eval_videos/{run}_b*_traj.npz"))):
    z = np.load(f); n += 1; k += bool(score(z["obj_pos"], z["obj_quat"], z["root_pos"], z["root_quat"], z["wrist"], box_lo=lo, box_hi=hi)["success"])
p = k / max(n, 1); zz = 1.96; d = 1 + zz * zz / max(n, 1); c = (p + zz * zz / (2 * max(n, 1))) / d; h = zz * math.sqrt(p * (1 - p) / max(n, 1) + zz * zz / (4 * max(n, 1) ** 2)) / d
print(f"  === {run} GR00T eval: {k}/{n} = {100*p:.1f}% (95% CI {100*(c-h):.0f}-{100*(c+h):.0f}%)")
PY
  python "$HOME/kevin/wt/dagger-watch-sparse/cluster/mpjpe_report.py" "$OUTV/${RUN}_b*_traj.npz" --output "$OUTV/${RUN}_mpjpe.json" >> "$ST" 2>&1 || log "MPJPE unavailable for $RUN; inspect metric report"
  if [ "${MONTAGE:-1}" = 1 ]; then   # MONTAGE=0: skip the 8 preview renders (scoring above is unaffected)
  local OUT=$OUTV/${RUN}_montage; rm -rf $OUT; mkdir -p $OUT
  for m in 0 12 25 37 50 62 75 87; do
    [ $m -lt $NEV ] || continue; echo $m > $W/_m$m.txt
    j=$((j + 1)); CUDA_VISIBLE_DEVICES=${G[$(( (j - 1) % NG ))]} python $RSL/play_vla_sonic.py --headless $XARGS --sonic-pt $PT --ref-motions-path $EVREFS --waist-dof 29 \
      --vla-checkpoint $VLA --motions-from $W/_m$m.txt --num-episodes 1 --seed $((300 + m)) --record-video $OUT/m$m --traj-dump $OUT/m$m > $W/mont_${RUN}_$m.log 2>&1 &
    [ $((j % NG)) -eq 0 ] && wait     # one render per slot at a time
  done; wait
  ls $OUT/*third_person.mp4 > $W/_mont_list.txt 2>/dev/null
  if [ -s $W/_mont_list.txt ]; then
    ffmpeg -v error -y $(sed 's/^/-i /' $W/_mont_list.txt | tr '\n' ' ') -filter_complex "$(n=$(wc -l < $W/_mont_list.txt); for i in $(seq 0 $((n-1))); do printf "[%d:v]scale=480:360[v%d];" $i $i; done; for i in $(seq 0 $((n-1))); do printf "[v%d]" $i; done; echo "xstack=inputs=$n:layout=$(python -c "n=$(wc -l < $W/_mont_list.txt);print('|'.join(f'{(i%4)*480}_{(i//4)*360}' for i in range(n)))"):fill=black[out]")" \
      -map "[out]" -c:v libx264 -crf 24 -pix_fmt yuv420p $OUT/${RUN}_montage.mp4 && log "  montage -> $OUT/${RUN}_montage.mp4"
  fi
  fi
  touch $W/ev_$RUN.done
}

[ "${DLINE_LIB:-0}" = 1 ] && return 0
# Independent eval workers can share the same validated run configuration.
if [ -n "${EVAL_ONLY:-}" ]; then
  evaluate "$EVAL_ONLY"
  exit $?
fi
# ---------------- base data ----------------
if [ ! -f $W/base.done ]; then
  if [ "$BASE" = collect ]; then
    log "base collection: expert over all $NREF clips of $REFS ($NSH shards)"
    collect $DS/${TAG}_demos 0
    BASEPAT="$DS/${TAG}_demos/**/*.hdf5"
  else
    BASEPAT=""; cp ${BASE#keep:} $W/keep_base_src.txt
  fi
  if [ "$BOX" = calibrate ]; then
    ( cd $RSL && python -m vla_sonic.calibrate_box --out $W/box.json $DS/${TAG}_demos >> $ST 2>&1 ) || die "box calibration failed"
  else cp $BOX $W/box.json; fi
  log "box: $(cat $W/box.json | tr -d '\n ' | cut -c1-120)"
  if [ -n "$BASEPAT" ]; then score_keep "$BASEPAT" $W/keep_base.txt base; else cp $W/keep_base_src.txt $W/keep_base.txt; fi
  [ $(grep -c . $W/keep_base.txt) -ge 40 ] || die "too few base episodes"
  touch $W/base.done
fi
DEMO_ROOT=$W/demo_root; rm -rf $DEMO_ROOT; mkdir -p $DEMO_ROOT; i=0
while read -r f; do [ -n "$f" ] && ln -s "$f" $DEMO_ROOT/$(basename "$f"); done < $W/keep_base.txt   # motion ids DAgger sweeps

if [ ! -f $W/r0.done ]; then
  D=$(build_ds base $W/keep_base.txt | tail -1); finetune $D ${TAG}_run01; touch $W/r0.done
fi
[ "${START_ROUND:-1}" -le 1 ] && evaluate ${TAG}_run01
KEEPS="$W/keep_base.txt"; STU=${TAG}_run01
for r in $(seq 1 $NR); do
  RUN=${TAG}_dagger${r}_run01
  if [ ! -f $W/r$r.done ]; then
    if [ ! -f $W/r${r}_collect.done ]; then
      log "DAgger round $r: student $STU, expert $CK on the paired reference, seed $((6 + r))"
      DR=$DEMO_ROOT; MOTION_ARGS=""
      if [ "${DAGGER_SAMPLE_ALL:-0}" = 1 ]; then
        python -c "import random; ids=sorted(random.Random(1000+$r).sample(range($NREF), min($DAGGER_N,$NREF))); print(' '.join(map(str,ids)))" > "$W/motion_sample_r$r.txt"
        MOTION_ARGS="--dagger-motion-ids $W/motion_sample_r$r.txt"
        log "  round $r: random reference IDs sampled from all $NREF references; manifest $W/motion_sample_r$r.txt"
      elif [ "$DAGGER_N" -gt 0 ] && [ $(grep -c . $W/keep_base.txt) -gt $DAGGER_N ]; then
        DR=$W/demo_root_r$r; rm -rf $DR; mkdir -p $DR
        python $HOME/kevin/dline/_dagger_sample.py demos $W/keep_base.txt $DAGGER_N $r > $W/demo_sample_r$r.txt
        while read -r f; do [ -n "$f" ] && ln -s "$f" $DR/$(basename "$f"); done < $W/demo_sample_r$r.txt
        log "  round $r: collecting on $DAGGER_N of $(grep -c . $W/keep_base.txt) reference motions (seed $((1000 + r)))"
      fi
      collect $DS/${TAG}_dagger$r $((6 + r)) --dagger-vla-checkpoint $HOME/kevin/checkpoints/$STU/checkpoint-$(steps_of $STU) --dagger-demo-root $DR $MOTION_ARGS \
        --dagger-chunk 8 --dagger-rescue --dagger-rescue-box file:$W/box.json --dagger-rescue-k-frac 1.0 --dagger-rescue-blend 0 \
        --dagger-expert-base reference --dagger-rollouts 1
      log "  round $r collected $(find $DS/${TAG}_dagger$r -name '*.hdf5' -not -name '_passA*' | wc -l) episodes ($(cat $DS/${TAG}_dagger$r/_log_*.txt | grep -c 'WROTE student') student successes)"
      touch $W/r${r}_collect.done
    fi
    score_keep "$DS/${TAG}_dagger$r/**/*.hdf5" $W/keep_d$r.txt dagger$r
    if [ "$DAGGER_N" -gt 0 ] && [ $(grep -c . $W/keep_d$r.txt) -gt $DAGGER_N ]; then
      python $HOME/kevin/dline/_dagger_sample.py cap $W/keep_d$r.txt $DAGGER_N $r
      log "  round $r: kept DAgger episodes capped at $DAGGER_N (random, seed $((2000 + r)))"
    fi
    KEEPS="$KEEPS $W/keep_d$r.txt"
    D=$(build_ds ds_dagger${r} $KEEPS | tail -1); finetune $D $RUN; touch $W/r$r.done
  else
    KEEPS="$KEEPS $W/keep_d$r.txt"
  fi
  # The next round's collection needs only this round's student, not its score: evaluate in the background and
  # move on (wait a few minutes so the eval's GPU memory is visible before collection picks its slots).
  if [ "$r" -ge "${START_ROUND:-1}" ]; then evaluate $RUN & sleep 240; fi; STU=$RUN
done
wait
log "LINE DONE: $(grep -E '=== .* GR00T eval' $ST | sed 's/.*=== //' | tr '\n' ';')"
touch $W/done
