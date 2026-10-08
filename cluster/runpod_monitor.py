"""Windows supervisor: queues work, returns verified checkpoints, records spend.

No cloud API credentials are needed for SSH. Pod termination remains a control-
plane action; status.json explicitly reports idle Pods and budget warnings.
"""
from pathlib import Path
import datetime
import json
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2] / 'out/runpod'
STAGE = Path('K:/_ftstage')
BL = 'sastrygrp-dvij@bluesclues.ist.berkeley.edu'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
INITIAL = {
    'af60v8f57_dagger2_run01': ('af60v8f57', 'af60v8f_ds_dagger2/lerobot/ds', 127000),
    'af60v8f57_dagger1_run01': ('af60v8f57', 'af60v8f_ds_dagger1/lerobot/ds', 97000),
}


def call(args, data=None, timeout=45):
    r = subprocess.run(args, input=data, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout, creationflags=FLAGS)
    if r.returncode:
        raise RuntimeError(f'Command failed ({r.returncode}): {r.stderr[-1500:]}')
    return r.stdout


def blue(command):
    return call(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', BL, command])


def podssh(pod, command, data=None):
    return call(['ssh', '-i', str(Path.home()/'.ssh/runpod_deadline'), '-o', 'BatchMode=yes',
                 '-o', 'ConnectTimeout=20', '-p', str(pod['port']), 'root@'+pod['host'], command], data)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


statefile = ROOT/'monitor_state.json'
state = json.loads(statefile.read_text()) if statefile.exists() else {'deliveries': {}, 'queued': {}}
while not (ROOT/'stop').exists():
    config = json.loads((ROOT/'pods.json').read_text())
    now = time.time()
    report = {'time': now, 'pods': [], 'estimated_spend': 0}
    requests = []
    lines = blue("for L in af60v8f57; do for f in ~/kevin/dline/$L/ftq/*.req; do test -f \"$f\" || continue; r=${f##*/}; r=${r%.req}; test -f ~/kevin/dline/$L/ftq/$r.ready || echo $L $r $(cat \"$f\"); done; done")
    for line in lines.splitlines():
        tag, run, dataset, steps = line.split()
        if '_dagger' in run and run not in INITIAL:
            requests.append({'run': run, 'line': tag, 'source': dataset, 'steps': int(steps), 'stage': 'epoch57_future'})
    for pod in config['pods']:
        report['estimated_spend'] += pod.get('accrued_cost', 0)
        if pod.get('terminated'):
            continue
        if pod.get('paused'):
            report['pods'].append({'id':pod['id'], 'state':'paused'})
            continue
        started = datetime.datetime.fromisoformat(pod['started'].replace('Z','+00:00')).timestamp()
        report['estimated_spend'] += (now-started)/3600*(pod['cost']+0.05)
        if pod['role'] == 'collection':
            report['pods'].append({'id':pod['id'], 'state':'collection_preparing'})
            continue
        code = "import json,pathlib; p=pathlib.Path('/workspace/queue'); print(json.dumps({'current':json.loads((p/'current.json').read_text()) if (p/'current.json').exists() else None,'pending':len(list(p.glob('*.job.json'))),'results':[json.loads(f.read_text()) for f in p.glob('*.result.json')],'ready':pathlib.Path('/workspace/logs/prepare.ready').exists()}))"
        status = json.loads(podssh(pod, 'python3 -c '+shlex.quote(code)))
        status['id'] = pod['id']
        for result in status['results']:
            run = result['run']
            if result['state'] != 'done':
                raise RuntimeError(f'Cloud training failed: {run}')
            delivery = state['deliveries'].get(run)
            if delivery is None:
                cmd = 'bash ~/kevin/wt/dagger-watch-sparse/cluster/runpod_receive.sh '+ ' '.join(shlex.quote(str(x)) for x in [pod['host'],pod['port'],run,result['steps'],result['line']])
                blue('export XLA_PYTHON_CLIENT_PREALLOCATE=false; tmux new-session -d -s '+run+'_receive_kevin '+shlex.quote(cmd+' > ~/kevin/runpod_transport/'+run+'.receive.log 2>&1'))
                state['deliveries'][run] = 'transferring'
            elif delivery == 'transferring':
                ready = blue('test ! -f ~/kevin/runpod_transport/received/'+run+' || cat ~/kevin/runpod_transport/received/'+run).strip()
                if ready:
                    directory = STAGE/result['stage']/run
                    directory.mkdir(parents=True,exist_ok=True)
                    (directory/'line').write_text(result['line'])
                    (directory/'steps').write_text(str(result['steps']))
                    (directory/'phase').write_text('done')
                    if result['stage'] == 'epoch57':
                        blue('python3 ~/kevin/wt/dagger-watch-sparse/cluster/bootstrap_epoch57.py '+shlex.quote(run))
                        (directory/'bootstrap.done').touch()
                    jobfile = directory/'adjob'
                    if not (ROOT/'adroit_unavailable').exists() and jobfile.exists() and jobfile.read_text().strip().isdigit():
                        j = jobfile.read_text().strip()
                        try:
                            call(['wsl.exe','-e','bash','-lc','ssh -o ControlPath=~/.ssh/adroit.sock -o ControlMaster=no -o BatchMode=yes adroit '+shlex.quote('scancel '+j)])
                            jobfile.write_text('none')
                        except (RuntimeError, subprocess.TimeoutExpired) as error:
                            (ROOT/'adroit_unavailable').write_text(str(error))
                    state['deliveries'][run] = 'delivered'
        running = status['current'] and status['current']['state']=='running'
        if status['ready'] and not running and status['pending']==0:
            candidates = list(requests)
            for run,(tag,relative,steps) in INITIAL.items():
                if pod.get('critical_only'):
                    continue
                directory = STAGE/'epoch57'/run
                if (directory/'phase').exists() and (directory/'phase').read_text().strip()=='done':
                    continue
                candidates.append({'run':run,'line':tag,'source':'~/kevin/datasets/'+relative,'relative':relative,'steps':steps,'stage':'epoch57'})
            candidates = [j for j in candidates if j['run'] not in state['queued'] and not any(r['run']==j['run'] for r in status['results'])]
            candidates.sort(key=lambda j:(0 if j['stage']=='epoch57_future' and j['line'].startswith('af60' if pod['role']=='pick' else 'T20') else 1,j['steps']))
            if candidates:
                job = candidates[0]
                if 'relative' in job:
                    job['dataset'] = '/root/kevin/datasets/'+job['relative']
                else:
                    destination = '/workspace/kevin/datasets/cloud_'+job['run']
                    source = job['source']
                    transfer = 'set -o pipefail; tar -C '+shlex.quote(source) + ' -cf - . | ssh -i ~/kevin/runpod_transport/key -o BatchMode=yes -p '+str(pod['port'])+' root@'+pod['host']+' '+shlex.quote('mkdir -p '+destination+'; tar --no-same-owner -C '+destination+' -xf -')
                    call(['ssh',BL,transfer],timeout=240)
                    job['dataset'] = destination
                podssh(pod,'cat > /workspace/queue/10-'+job['run']+'.job.json',json.dumps(job))
                state['queued'][job['run']] = pod['id']
                status['queued_next'] = job['run']
            else:
                status['idle_needs_review'] = True
        report['pods'].append(status)
    report['budget_warning'] = report['estimated_spend'] >= 85
    save(statefile,state)
    save(ROOT/'status.json',report)
    print(json.dumps(report),flush=True)
    time.sleep(30)
