#!/bin/bash
# Resume the explicitly approved interrupted job once its exact env is restored.
set -euo pipefail
cd /workspace/kevin/DreamVLA_runtime
until test -f /workspace/logs/recovery_environment.ready; do sleep 5; done
python3 - <<'PY'
import json, shutil, os
from pathlib import Path
q=Path('/workspace/queue')
j=json.loads((q/'current.json').read_text())
assert j['state']=='recovering', j
run=j['run']
root=Path('/workspace/kevin/checkpoints')/run
checkpoints=sorted(root.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
valid=[]
for p in checkpoints:
    index=p/'model.safetensors.index.json'
    if not index.exists() or not (p/'trainer_state.json').exists(): continue
    files=set(json.loads(index.read_text())['weight_map'].values())
    if all((p/f).exists() and (p/f).stat().st_size>0 for f in files):valid.append(p)
assert valid, 'No complete checkpoint; will not restart from scratch'
ck=valid[0]
state=json.loads((ck/'trainer_state.json').read_text())
assert 0<state['global_step']<j['steps']
assert not (ck/'optimizer.pt').exists(), 'Use normal resume for full checkpoints'
backup=Path('/workspace/recovery_originals')/run/ck.name
if not backup.exists():
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ck,backup,copy_function=os.link)
for k in ['started','pid','state','finished','exit_code','error_log_tail']:
    j.pop(k,None)
j['resume_checkpoint']=str(ck)
pending=q/('00-recover-'+run+'.job.json')
assert not pending.exists(), 'Recovery already queued'
tmp=pending.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(pending)
print('RECOVERY_QUEUED',run,state['global_step'],j['steps'],flush=True)
PY
touch /workspace/logs/prepare.ready
exec python3 cluster/runpod_queue.py
