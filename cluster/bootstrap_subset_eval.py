"""Launch a subset base checkpoint's comparable standard eval and MPJPE recording."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
p=argparse.ArgumentParser(); p.add_argument('run',choices=['pick10','pick50','walk10','walk50']); a=p.parse_args()
root=Path.home()/'kevin'; work=root/'dline'/a.run
oldtag='af60v8f' if a.run.startswith('pick') else 'T20maxf'
env=dict(json.loads((root/'dline/watch/state.json').read_text())[oldtag]['env'])
code=Path(env.get('CODE',str(root/'DreamVLA')))
shutil.copy2(root/'dline'/oldtag/'box.json',work/'box.json')
env.update(TAG=a.run,EVAL_ONLY=a.run,BASE='keep:'+str(root/'dline'/oldtag/'keep_base.txt'),
           SKIP_CODE_UPDATE='1',MONTAGE='0',EVAL_SCRIPT=str(root/'wt/dagger-watch-sparse/Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'),
           PYTHONPATH=str(code/'Training/scripts/reinforcement_learning/rsl_rl'))
if oldtag=='T20maxf':
    env['ENVX']='HS_REWORK_CLOSE_ON_ARRIVAL=1 HS_REWORK_PALM_NORMAL=0 HS_REWORK_PALM_REACH=0.07 HS_PALM_NORMAL_SIGN=+1'
session='eval_'+a.run+'_kevin'
if not (work/('ev_'+a.run+'.done')).exists() and subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:
    command='export XLA_PYTHON_CLIENT_PREALLOCATE=false; unset HS_REWORK_ARRIVE_FINGER_D; '+' '.join(f'{k}={shlex.quote(v)}' for k,v in env.items())+' bash ~/kevin/dline/_dline.sh'
    subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
print('Eval gate ready',a.run)
