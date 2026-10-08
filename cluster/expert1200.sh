#!/bin/bash
# One attempt per reference per pass; archive handshake bounds scratch usage.
set -u
DLINE_LIB=1 source ~/kevin/dline/_dline.sh
cp "$EXPERT_BOX" "$W/box.json"
TOTAL=0
for pass in $(seq 0 $((EXPERT_PASSES-1))); do
  count=$EXPERT_REFERENCE_COUNT
  [ "$pass" -eq $((EXPERT_PASSES-1)) ] && count=$EXPERT_LAST_COUNT
  out="$DS/${TAG}_pass$pass"
  if [ -f "$W/pass$pass.archived" ]; then TOTAL=$((TOTAL+count)); continue; fi
  if [ -e "$out" ] && [ ! -f "$W/pass$pass.ready" ]; then die "partial pass $pass requires attempt-ledger review before restart"; fi
  if [ ! -f "$W/pass$pass.ready" ]; then
    NREF=$count
    collect "$out" $((42000+pass)) || die "expert pass $pass collector failed; review attempts before retry"
    observed=$(grep -hE '^\[INFO\] motion [0-9]+/' "$out"/_log_*.txt | wc -l)
    [ "$observed" -eq "$count" ] || die "expected $count attempts, observed $observed in pass $pass"
    score_keep "$out/**/*.hdf5" "$out/keep.txt" "expert pass $pass"
    printf '%s %s\n' "$count" "$out" > "$W/pass$pass.ready"
  fi
  TOTAL=$((TOTAL+count))
  log "expert pass $pass completed; cumulative attempts $TOTAL; waiting for verified Adroit archive"
  while [ ! -f "$W/pass$pass.archived" ]; do sleep 30; done
done
[ "$TOTAL" -eq 1200 ] || die "expert attempt budget mismatch $TOTAL"
log "DONE: 1200 expert attempts, passing manifests archived"
touch "$W/done"
