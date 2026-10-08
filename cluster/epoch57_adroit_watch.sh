#!/bin/bash
# Independent batch: never modifies the running legacy orchestrator.
set -u
ORCH_LIB=1 source /k/_ftstage/_ftorch.sh
ad(){ python '/k/Coding Projects/Labs/EMBER/WBCBenchmark/cluster/adroit_bridge.py'; }
STAGE=/k/_ftstage/epoch57
WSTAGE=/mnt/k/_ftstage/epoch57
LOG=$STAGE/watch.log
mkdir -p "$STAGE"
mkdir "$STAGE/watch.lockdir" 2>/dev/null || exit 0
trap 'rmdir "$STAGE/watch.lockdir" 2>/dev/null' EXIT
echo $$ > "$STAGE/watch.pid"
while [ ! -f "$STAGE/stop" ]; do
  remaining=0
  for r in T20maxf57_dagger2_run01 af60v8f57_dagger2_run01 T20maxf57_run01 af60v8f57_dagger1_run01 T20maxf57_dagger1_run01 af60v8f57_run01; do
    directory="$STAGE/$r"
    [ -f "$directory/adjob" ] || continue
    r=${directory##*/}
    phase=$(get "$r" phase)
    if [ "$phase" = done ]; then
      if [ ! -f "$directory/bootstrap.done" ]; then
        if bl "python3 ~/kevin/wt/dagger-watch-sparse/cluster/bootstrap_epoch57.py $r" >> "$LOG" 2>&1; then touch "$directory/bootstrap.done"; else remaining=$((remaining+1)); fi
      fi
      continue
    fi
    remaining=$((remaining+1))
    # Opportunistic bluesclues runs race the queued Adroit copy. Deliver only
    # after the local trainer exits and its final checkpoint exists.
    if [ "$(get "$r" primary)" = bluesclues ] && ! local_alive "$r" bluesclues; then
      if ck_on "$r" bluesclues; then
        deliver "$r" bluesclues
        continue
      elif [ $(( $(date +%s) - $(get "$r" started) )) -gt 300 ]; then
        put "$r" primary adroit
        rm -f "$directory/local.started"
        log "$r: local trainer exited without a checkpoint; restored queued fallback and eligibility"
      fi
    fi
    j=$(get "$r" adjob)
    result=$(printf 'squeue -j %s -h -o %%T\n' "$j" | ad 2>>"$LOG") || { log "Adroit connection failed; stopping for user intervention"; exit 1; }
    state=$(echo "$result" | tr -d '\r\n ')
    put "$r" state "${state:-LEFT_QUEUE}"
    # Prioritize round 2 by directory order below when capacity returns. Do not
    # leave queued restarts dependent on a manual opportunistic launch.
    if [ "$state" = PENDING ] && [ "$(get "$r" primary)" = adroit ] && [ ! -f "$directory/local.started" ]; then
      free=$(bl "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | awk '$1<2000{n++} END{print n+0}')
      if [ "$free" -ge 2 ]; then
        case "$r" in
          af60v8f57_dagger1_run01) ds='~/kevin/datasets/af60v8f_ds_dagger1/lerobot/ds' ;;
          af60v8f57_dagger2_run01) ds='~/kevin/datasets/af60v8f_ds_dagger2/lerobot/ds' ;;
          T20maxf57_run01) ds='~/kevin/datasets/T20maxf_base/lerobot/ds' ;;
          T20maxf57_dagger1_run01) ds='~/kevin/datasets/T20maxf_ds_dagger1/lerobot/ds' ;;
          T20maxf57_dagger2_run01) ds='~/kevin/datasets/T20maxf_ds_dagger2/lerobot/ds' ;;
          *) ds='' ;;
        esac
        if [ -n "$ds" ] && bl "test -f $ds/meta/info.json && { test ! -f ~/kevin/checkpoints/_ftlogs/$r.log || cp ~/kevin/checkpoints/_ftlogs/$r.log ~/kevin/checkpoints/_ftlogs/$r.previous.\$(date +%s).log; } && tmux new-window -d -t train_kevin -n $r 'export XLA_PYTHON_CLIENT_PREALLOCATE=false; bash ~/kevin/wt/dagger-watch-sparse/cluster/run_gr00t_reserved.sh $ds $r $(get "$r" steps) > ~/kevin/checkpoints/_ftlogs/$r.log 2>&1'"; then
          put "$r" primary bluesclues
          put "$r" started "$(date +%s)"
          touch "$directory/local.started"
          log "$r: opportunistic bluesclues launch; Adroit retained until a checkpoint wins"
        fi
      fi
    fi
    if [ -z "$state" ]; then
      if ck_on "$r" adroit; then
        log "$r: final checkpoint present; delivering"
        deliver "$r" adroit
      else
        result=$(printf 'sacct -j %s -n -X -o State,ExitCode\n' "$j" | ad 2>>"$LOG") || exit 1
        put "$r" accounting "$result"
        log "$r: left queue without final checkpoint: $result"
      fi
    fi
  done
  [ "$remaining" -eq 0 ] && { log "All six checkpoints delivered"; exit 0; }
  sleep 60
done
