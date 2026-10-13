"""Serve explicit Pick50 fine-tune requests on Biped and deliver verified weights.

Runs on Windows. An SSH failure stops this bridge for user troubleshooting;
it never blindly restarts or cancels a remote training job.
"""
import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from types import SimpleNamespace

from stream_directory import ssh, transfer

TAG='pick50_dagger57_20261013'
OUT=Path(__file__).resolve().parents[2]/'out'
STATE=OUT/'pick_budget_biped_bridge.json'


def remote(host,code):
    p=subprocess.run(ssh(host,'python3 -'),input=code,capture_output=True,text=True,timeout=45,
                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if p.returncode:raise RuntimeError(host+': '+p.stderr[-4000:]+p.stdout[-1000:])
    return json.loads(p.stdout)


def state_write(state):
    state['observed']=time.time();state['bridge_pid']=os.getpid()
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)


def pending():
    return remote('bluesclues',r'''
from pathlib import Path
import json
r=Path.home()/'kevin';w=r/'dline/pick50_dagger57_20261013/external'
rows=[]
for p in sorted(w.glob('finetune_*.request.json')):
 if not p.with_name(p.name.replace('.request.json','.reply.json')).exists():rows.append(json.loads(p.read_text()))
print(json.dumps({'root':str(r),'requests':rows}))
''')


def start_job(request,blue_root):
    assert request['host']=='biped' and request['global_batch_size']==16 and request['passes']==5.7
    assert request['steps']==math.ceil(request['frames']*57/160)
    run=request['run'];assert run in [TAG+'_r2',TAG+'_r3']
    dataset=str(PurePosixPath(request['dataset']).relative_to(blue_root))
    checkpoint=str(PurePosixPath(request['checkpoint']).relative_to(blue_root))
    capacity=remote('biped',r'''
import subprocess,json
s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
print(json.dumps({'available':all(int(row.split(',')[1])<2000 for row in s.splitlines()),'rows':s}))
''')
    if not capacity['available']:return None
    transfer(SimpleNamespace(source_host='bluesclues',destination_host='biped',source=dataset,
                             destination=dataset,inference=False,receipt=OUT/(run+'_dataset_transfer.json')))
    payload=dict(request,dataset_relative=dataset,checkpoint_relative=checkpoint)
    launch=r'''
from pathlib import Path
import json,subprocess,shlex,time
p=PAYLOAD;r=Path.home()/'kevin';run=p['run'];ds=r/p['dataset_relative']
i=json.loads((ds/'meta/info.json').read_text());assert i['total_frames']==p['frames'] and i['total_episodes']==p['episodes']
s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert len(s.splitlines())==2 and all(int(x)<2000 for x in s.splitlines()),'Biped capacity changed; no training launched'
w=r/'dline'/run;w.mkdir(parents=True,exist_ok=True)
assert not (w/'job.json').exists() and not (r/p['checkpoint_relative']).exists()
session='train_gr00t'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:
 subprocess.run(['tmux','new-session','-d','-s',session,'export XLA_PYTHON_CLIENT_PREALLOCATE=false; exec bash'],check=True)
command='export XLA_PYTHON_CLIENT_PREALLOCATE=false; bash '+shlex.quote(str(r/'DreamVLA/cluster/biped_pick_budget_ft.sh'))+' '+shlex.quote(str(ds))+' '+shlex.quote(run)+' '+str(p['steps'])+' > '+shlex.quote(str(w/'training.log'))+' 2>&1'
subprocess.run(['tmux','new-window','-d','-t',session,'-n',run,command],check=True)
job=dict(p,started=time.time(),session=session,training_log=str(w/'training.log'))
(w/'job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job))
'''.replace('PAYLOAD',repr(payload))
    return remote('biped',launch)


def job_status(job):
    return remote('biped',r'''
from pathlib import Path
import json,re
r=Path.home()/'kevin';run=RUN;w=r/'dline'/run
exitfile=w/'exit_code.txt';pidfile=w/'worker.pid'
out={'exit_code':int(exitfile.read_text()) if exitfile.exists() else None,'pid':int(pidfile.read_text()) if pidfile.exists() else None}
if out['pid']:
 try:out['args']=(Path('/proc')/str(out['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:out['args']=''
log=w/'training.log'
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-24000));text=f.read().decode(errors='replace')
 out['log_tail']=text[-3000:];out['log_age_seconds']=__import__('time').time()-log.stat().st_mtime
if out['exit_code']==0:
 p=PAYLOAD;ck=r/p['checkpoint_relative'];d=json.loads((ck/'trainer_state.json').read_text())
 assert d['global_step']==p['steps']
 cfg=(ck/'experiment_cfg/config.yaml').read_text()
 for key,value in [('global_batch_size','16'),('num_gpus','2'),('gradient_accumulation_steps','1'),('bf16','true')]:
  assert re.search(r'^  '+key+': '+value+r'$',cfg,re.M),key
 out['verified_final_step']=d['global_step']
print(json.dumps(out))
'''.replace('RUN',repr(job['run'])).replace('PAYLOAD',repr(job)))


def deliver(job):
    run=job['run'];ck=job['checkpoint_relative']
    transfer(SimpleNamespace(source_host='biped',destination_host='bluesclues',source=ck,destination=ck,
                             inference=True,receipt=OUT/(run+'_checkpoint_transfer.json')))
    reply={'state':'completed','host':'biped','run':run,'steps':job['steps'],'time':time.time(),
           'global_batch_size':16,'num_gpus':2,'checkpoint_relative':ck}
    remote('bluesclues',r'''
from pathlib import Path
import json
r=Path.home()/'kevin';p=PAYLOAD;ck=r/p['checkpoint_relative']
assert json.loads((ck/'trainer_state.json').read_text())['global_step']==p['steps']
w=r/'dline/pick50_dagger57_20261013/external';f=w/('finetune_'+p['run']+'.reply.json')
assert not f.exists();tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(p,indent=2));tmp.replace(f)
print(json.dumps(p))
'''.replace('PAYLOAD',repr(reply)))


def main():
    OUT.mkdir(exist_ok=True)
    # A native file lock prevents two bridges from serving the same request.
    lock=(OUT/'pick_budget_biped_bridge.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'phase':'idle','delivered':[]}
    if state['phase'] not in ['idle','running','complete']:raise RuntimeError('Prior bridge error needs diagnosis: '+state['phase'])
    try:
        while state['phase']!='complete':
            if state['phase']=='idle':
                rows=pending()
                if rows['requests']:
                    state.update(phase='staging_dataset',request=rows['requests'][0]);state_write(state)
                    job=start_job(rows['requests'][0],rows['root'])
                    if job:state.update(phase='running',job=job)
                    else:state['phase']='idle'
            elif state['phase']=='running':
                result=job_status(state['job']);state['last_job_status']=result
                missing=result['exit_code'] is None and result.get('pid') and not result.get('args')
                state['missing_worker_observations']=state.get('missing_worker_observations',0)+1 if missing else 0
                assert state['missing_worker_observations']<2,'Training supervisor disappeared without a terminal exit record'
                if result['exit_code'] is not None:
                    assert result['exit_code']==0,result
                    state['phase']='delivering';state_write(state);deliver(state['job'])
                    state['delivered'].append(state['job']['run'])
                    state['phase']='complete' if state['job']['run'].endswith('_r3') else 'idle'
            state_write(state)
            print(json.dumps({'phase':state['phase'],'run':state.get('job',{}).get('run'),'time':state['observed']}),flush=True)
            if state['phase']!='complete':time.sleep(30)
    except BaseException as e:
        state['phase']='error';state['error']=repr(e);state_write(state);raise


if __name__=='__main__':main()
