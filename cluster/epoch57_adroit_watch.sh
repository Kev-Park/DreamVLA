#!/bin/bash
# Independent batch: never modifies the running legacy orchestrator.
set -u
ORCH_LIB=1 source /k/_ftstage/_ftorch.sh
STAGE=/k/_ftstage/epoch57
WSTAGE=/mnt/k/_ftstage/epoch57
LOG=$STAGE/watch.log
mkdir -p "$STAGE"
exec 9>"$STAGE/watch.lock"
flock -n 9 || exit 0
while [ ! -f "$STAGE/stop" ]; do
  remaining=0
  for directory in "$STAGE"/*; do
    [ -f "$directory/adjob" ] || continue
    r=${directory##*/}
    phase=$(get "$r" phase)
    [ "$phase" = done ] && continue
    remaining=$((remaining+1))
    j=$(get "$r" adjob)
    result=$(printf 'squeue -j %s -h -o %%T\n' "$j" | ad 2>>"$LOG") || { log "Adroit connection failed; stopping for user intervention"; exit 1; }
    state=$(echo "$result" | tr -d '\r\n ')
    put "$r" state "${state:-LEFT_QUEUE}"
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
