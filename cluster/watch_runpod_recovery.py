"""Recover our stranded R2 pod when its host has capacity; never retry SSH failures.

No new pod is allocated. Uses the existing budget guard and the approved
weights-only recovery recipe. Checks control-plane capacity once per minute.
"""
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time
import tomllib
import urllib.request

ROOT=Path(__file__).resolve().parents[2]/'out/runpod'
POD='feopiqtnufdbsl'
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)

def write(p,value):
    t=p.with_suffix('.recovery.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def api(path,body=None):
    key=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
    req=urllib.request.Request('https://v2-rest.runpod.io/v2/'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'ember-recovery'})
    with urllib.request.urlopen(req,timeout=25) as f:return json.load(f)

def capacity():
    key=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
    query='{ pod(input: {podId: "'+POD+'"}) { machine { gpuAvailable memoryTotal memoryReserved vcpuTotal vcpuReserved } } }'
    req=urllib.request.Request('https://api.runpod.io/graphql?api_key='+key,
        data=json.dumps({'query':query}).encode(),headers={'Content-Type':'application/json','User-Agent':'ember-recovery'})
    with urllib.request.urlopen(req,timeout=25) as f:data=json.load(f)
    return data['data']['pod']['machine']

def call(args,timeout=45):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,creationflags=FLAGS)
    if p.returncode:raise RuntimeError(p.stderr[-1500:])
    return p.stdout

def main():
    while True:
        config=json.loads((ROOT/'pods.json').read_text())
        guard=json.loads((ROOT/'budget_guard_status.json').read_text())
        deadline=datetime.datetime.fromisoformat(config['deadline'].replace('Z','+00:00')).timestamp()
        if time.time()>=deadline or guard['estimated_spend']>=config['budget']-20:
            raise RuntimeError('Insufficient remaining time/budget for automatic recovery')
        machine=capacity()
        write(ROOT/'r2_recovery.json',dict(time=time.time(),state='awaiting_host_capacity',machine=machine))
        if machine['gpuAvailable']>0 and machine['memoryTotal']-machine['memoryReserved']>=377 and machine['vcpuTotal']-machine['vcpuReserved']>=36:
            break
        time.sleep(60)
    started=api('pods/'+POD+'/action',{'action':'start'})
    config=json.loads((ROOT/'pods.json').read_text())
    pod=next(p for p in config['pods'] if p['id']==POD)
    pod.update(paused=False,started=started['startedAt'],provisioning_until=time.time()+2400)
    # The monitor must not use the stale SSH port before the runtime is ready.
    old_role=pod['role'];pod['role']='benchmark'
    write(ROOT/'pods.json',config)
    for _ in range(60):
        live=api('pods/'+POD)
        direct=(live.get('ssh') or {}).get('direct')
        if direct:break
        time.sleep(5)
    else:raise RuntimeError('No SSH runtime after five minutes')
    pod.update(host=direct['host'],port=direct['port'])
    write(ROOT/'pods.json',config)
    ssh=['ssh','-i',str(Path.home()/'.ssh/runpod_deadline'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=15','-p',str(pod['port']),'root@'+pod['host']]
    prep='import json; from pathlib import Path; p=Path("/workspace/queue/current.json"); j=json.loads(p.read_text()); j["state"]="recovering"; p.write_text(json.dumps(j)); Path("/workspace/logs/prepare.ready").unlink(missing_ok=True)'
    call(ssh+['python3 -c '+shlex.quote(prep)+' && git -C /workspace/kevin/DreamVLA_runtime pull --ff-only origin g1'])
    remote_ssh='ssh -i ~/kevin/runpod_transport/key -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p '+str(pod['port'])
    target='root@'+pod['host']
    transfer='set -euo pipefail; '+remote_ssh+' '+target+' '+shlex.quote('mkdir -p /opt/kevin/Isaac-GR00T/.venv')+'; rsync -az --partial --inplace --compress-choice=zstd --compress-level=1 -e '+shlex.quote(remote_ssh)+' ~/kevin/Isaac-GR00T/.venv/ '+target+':/opt/kevin/Isaac-GR00T/.venv/; '+remote_ssh+' '+target+' '+shlex.quote('bash /workspace/kevin/DreamVLA_runtime/cluster/runpod_prepare.sh && touch /workspace/logs/recovery_environment.ready')
    command='export XLA_PYTHON_CLIENT_PREALLOCATE=false; tmux new-session -d -s r2_recovery_env_kevin '+shlex.quote('bash -lc '+shlex.quote(transfer)+' > ~/kevin/runpod_transport/r2_recovery_env.log 2>&1')
    call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','sastrygrp-dvij@bluesclues.ist.berkeley.edu',command])
    call(ssh+['nohup bash /workspace/kevin/DreamVLA_runtime/cluster/runpod_recover.sh > /workspace/logs/recovery.log 2>&1 < /dev/null &'])
    config=json.loads((ROOT/'pods.json').read_text())
    next(p for p in config['pods'] if p['id']==POD)['role']=old_role
    write(ROOT/'pods.json',config)
    write(ROOT/'r2_recovery.json',dict(time=time.time(),state='environment_restoring_then_automatic_recovery',host=pod['host'],port=pod['port']))

if __name__=='__main__':
    try:main()
    except Exception as e:
        # Never print authenticated URLs from HTTP exception objects.
        write(ROOT/'r2_recovery.json',dict(time=time.time(),state='blocked',error=type(e).__name__,detail=str(e) if not isinstance(e,urllib.error.HTTPError) else 'HTTP '+str(e.code)))
        raise SystemExit(1)
