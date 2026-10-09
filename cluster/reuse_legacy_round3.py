"""Continue epoch57 round 3 using the explicitly requested earlier collection."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

r=Path.home()/'kevin'
old=r/'dline/af60v8f';w=r/'dline/af60v8f57'
source=r/'datasets/af60v8f_ds_dagger3'
info=json.loads((source/'lerobot/ds/meta/info.json').read_text())
assert (source/'built.ok').exists() and (old/'r3_collect.done').exists()
assert info['total_episodes']==892 and info['total_frames']==443502
for name in ['keep_base.txt','keep_d1.txt','keep_d2.txt']:
    assert (old/name).read_bytes()==(w/name).read_bytes(), name
keep=(old/'keep_d3.txt').read_text().splitlines()
assert len(keep)==181 and all(Path(p).exists() for p in keep)
assert not (w/'r3.done').exists(), 'Do not replace a trained round'
for dst,src in [(r/'datasets/af60v8f57_dagger3',r/'datasets/af60v8f_dagger3'),
                (r/'datasets/af60v8f57_ds_dagger3',source)]:
    if dst.exists():assert dst.resolve()==src.resolve(), str(dst)
    else:dst.symlink_to(src,target_is_directory=True)
shutil.copy2(old/'keep_d3.txt',w/'keep_d3.txt')
run='af60v8f57_dagger3_run01'
steps=math.ceil(info['total_frames']*5.7/64)
recipe=dict(global_batch_size=64,steps=steps,frames=info['total_frames'],
    target_passes=5.7,effective_passes=steps*64/info['total_frames'],
    precision='bf16',learning_rate=1e-4,optimizer='adamw_torch',
    reason='Previously approved batch options: batch64 balances deadline and remaining $160 total budget',
    collection_student='af60v8f_dagger2_run01/checkpoint-30000',
    collection_reused=True,retained_new_episodes=181,recorded_rollouts=196,
    source_dataset=str(source/'lerobot/ds'),
    keep_sha256=hashlib.sha256((old/'keep_d3.txt').read_bytes()).hexdigest(),
    selected_at=time.time())
target=w/('recipe_'+run+'.json')
if target.exists():assert json.loads(target.read_text())==recipe, 'Recipe already exists; inspect before changing'
target.write_text(json.dumps(recipe,indent=2))
(w/('steps_'+run)).write_text(str(steps)+'\n')
(w/'r3_collect.done').touch()
env=json.loads((w/'epoch57_environment.json').read_text())
assign=' '.join(f'{k}={shlex.quote(v)}' for k,v in env.items())
session='dline_v857_kevin'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0, 'Chain already active'
command='export XLA_PYTHON_CLIENT_PREALLOCATE=false; unset HS_REWORK_ARRIVE_FINGER_D; '+assign+' bash '+shlex.quote(str(r/'wt/dagger-watch-sparse/cluster/dagger_line.sh'))
subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
print(json.dumps(recipe,indent=2))
