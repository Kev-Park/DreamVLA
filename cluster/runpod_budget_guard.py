"""Independent control-plane cutoff; credentials stay on this Windows machine.

Stops (never terminates) only the four Pods created for this experiment. Stopping
retains persistent /workspace outputs. Estimated charges include storage reserve.
"""
import datetime
import json
import re
from pathlib import Path
import time
import tomllib
import urllib.request
import subprocess
import shlex
import os

ROOT = Path(__file__).resolve().parents[2] / 'out/runpod'
OWNED = {'04z8d0a941vb9z', 'pscjg26ij7t3yk', 'gop5085t076jhp', '48w7f0lwpz4dfu',
         'itrttaagys4vib', 'feopiqtnufdbsl', 'qv0an1x9jn918o'}


def api(pod_id, action=None):
    key = tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
    url = 'https://v2-rest.runpod.io/v2/pods/' + pod_id
    data = None if action is None else json.dumps({'action': action}).encode()
    request = urllib.request.Request(url + ('/action' if action else ''), data=data,
        headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json',
                 'User-Agent': 'ember-budget-monitor'})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def estimate(config, now):
    cost = 0
    for pod in config['pods']:
        cost += pod.get('accrued_cost', 0)
        if pod.get('terminated') or pod.get('paused'):
            continue
        start = datetime.datetime.fromisoformat(pod['started'].replace('Z','+00:00')).timestamp()
        cost += max(0, now-start)/3600*(pod['cost']+0.05)
    return cost


def capture_logs(pod):
    """Copy failure evidence off the host before releasing its GPU slot."""
    code = "import pathlib,json; p=pathlib.Path('/workspace/logs'); fs=sorted(p.glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:5]; print(json.dumps({f.name:f.read_bytes()[-24000:].decode('utf8','replace') for f in fs}))"
    result = subprocess.run(['ssh', '-i', str(Path.home()/'.ssh/runpod_deadline'),
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', str(pod['port']),
        'root@'+pod['host'], 'python3 -c '+shlex.quote(code)],
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])
    data = json.loads(result.stdout)
    (ROOT/('logs_'+pod['id']+'.json')).write_text(json.dumps(data, indent=2))
    return list(data)


def main():
    config = json.loads((ROOT/'pods.json').read_text())
    api(next(p['id'] for p in config['pods'] if not p.get('terminated')))
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    history = {}
    while not (ROOT/'budget_guard.stop').exists():
        try:
            config = json.loads((ROOT/'pods.json').read_text())
            now = time.time()
            spent = estimate(config, now)
            deadline = datetime.datetime.fromisoformat(config['deadline'].replace('Z','+00:00')).timestamp()
            if os.name == 'nt' and now >= deadline:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            cutoff = config['budget']-2
            status = {'time': now, 'authenticated': True, 'estimated_spend': spent,
                      'cutoff': cutoff, 'actions': [], 'activity': [], 'alerts': []}
            control = json.loads((ROOT/'budget_control.json').read_text()) if (ROOT/'budget_control.json').exists() else {}
            workload = json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
            deliveries = json.loads((ROOT/'monitor_state.json').read_text()).get('deliveries', {}) if (ROOT/'monitor_state.json').exists() else {}
            for pod in config['pods']:
                if pod['id'] not in OWNED or pod.get('terminated') or pod.get('paused'):
                    continue
                reason = ('budget cutoff' if spent >= cutoff else
                          'deadline' if now >= deadline else
                          control.get('release', {}).get(pod['id']))
                replacement = next((p for p in workload.get('pods', []) if p['id'] == 'pscjg26ij7t3yk'), {})
                if pod['id'] == '04z8d0a941vb9z' and pod.get('role') != 'benchmark' and replacement.get('training_step', 0) >= 1000:
                    reason = reason or 'two-H100 replacement passed 1000 training steps'
                if pod.get('benchmark_until', now+1) <= now:
                    reason = reason or 'bounded benchmark window ended'
                observed = next((p for p in workload.get('pods', []) if p['id'] == pod['id']), {})
                current = observed.get('current') or {}
                live = api(pod['id'])
                if live['status'] in ('EXITED', 'ERROR'):
                    reason = reason or 'provider reports stopped pod'
                previous = history.setdefault(pod['id'], {'step': None, 'progress_time': now})
                step = observed.get('training_step')
                if step != previous['step']:
                    previous.update(step=step, progress_time=now)
                gpus = (live.get('runtime') or {}).get('gpus', [])
                idle = bool(gpus) and all(g.get('util', 100) == 0 and g.get('memoryUtil', 100) == 0 for g in gpus)
                if idle:
                    previous.setdefault('idle_since', now)
                else:
                    previous.pop('idle_since', None)
                age = now-datetime.datetime.fromisoformat(pod['started'].replace('Z','+00:00')).timestamp()
                idle_seconds = now-previous.get('idle_since', now)
                status['activity'].append({'id':pod['id'], 'status':live['status'], 'gpus':gpus,
                    'run':current.get('run'), 'step':step, 'idle_seconds':idle_seconds,
                    'seconds_without_progress':now-previous['progress_time']})
                if (idle_seconds >= 300 and age >= 900 and now >= pod.get('provisioning_until', 0) and current.get('state') != 'running'
                        and observed.get('pending', 0) == 0
                        and (current.get('state') != 'done'
                             or deliveries.get(current.get('run')) == 'delivered')):
                    reason = reason or 'no GPU activity or allocation for five minutes'
                if current.get('state') == 'running' and now-previous['progress_time'] > 600:
                    status['alerts'].append({'id':pod['id'], 'reason':'training progress stalled for ten minutes'})
                if current.get('state') == 'failed':
                    previous.setdefault('failed_since', now)
                    if not previous.get('logs_captured') and not previous.get('log_capture_failed'):
                        try:
                            previous['logs_captured'] = capture_logs(pod)
                        except Exception as error:
                            previous['log_capture_failed'] = str(error)
                    status['alerts'].append({'id':pod['id'], 'reason':'training failed',
                        'logs_captured':previous.get('logs_captured'), 'capture_error':previous.get('log_capture_failed')})
                    if reason not in ('budget cutoff', 'deadline') and now-previous['failed_since'] < 600:
                        # Keep a short diagnosis window; stopping can strand host-local logs.
                        reason = None
                    elif not reason:
                        reason = 'failed workload diagnosis window ended'
                match = re.search(r'_dagger(\d+)_', current.get('run', ''))
                if (pod.get('role') == 'baseline1200' and current.get('state') == 'done'
                        and deliveries.get(current['run']) == 'delivered'):
                    reason = reason or 'baseline checkpoint verified and delivered'
                if (current.get('state') == 'done' and match
                        and int(match.group(1)) >= pod.get('max_round', 3)
                        and deliveries.get(current['run']) == 'delivered'
                        and observed.get('pending', 0) == 0):
                    reason = reason or 'final cloud checkpoint verified and delivered'
                if not reason:
                    continue
                if not previous.get('logs_captured') and not previous.get('log_capture_failed'):
                    try:
                        previous['logs_captured'] = capture_logs(pod)
                    except Exception as error:
                        previous['log_capture_failed'] = str(error)
                if live['status'] not in ('EXITED', 'ERROR'):
                    api(pod['id'], 'stop')
                event = {'id': pod['id'], 'reason': reason, 'time': now}
                # Separate state avoids racing the workload monitor's manifest.
                (ROOT/('stopped_'+pod['id']+'.json')).write_text(json.dumps(event))
                status['actions'].append(event)
                pod['paused'] = True
                start = datetime.datetime.fromisoformat(pod['started'].replace('Z','+00:00')).timestamp()
                pod['accrued_cost'] = pod.get('accrued_cost', 0)+max(0,now-start)/3600*(pod['cost']+0.05)
                temp = ROOT/'pods.guard.tmp'
                temp.write_text(json.dumps(config, indent=2))
                temp.replace(ROOT/'pods.json')
            (ROOT/'budget_guard_status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
        except Exception as error:
            # HTTP errors do not contain the Authorization header; never print requests.
            print(json.dumps({'time':time.time(), 'error':str(error)}), flush=True)
        time.sleep(15)


if __name__ == '__main__':
    main()
