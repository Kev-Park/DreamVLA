"""Bounded B200 production rollout; preserve H100 training until it catches up."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2] / 'out/runpod'
NEW, OLD = 'itrttaagys4vib', 'pscjg26ij7t3yk'
RUN = 'af60v8f57_dagger2_run01'


def save(path, data):
    temporary = path.with_suffix('.handoff.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def release(pod, reason):
    path = ROOT/'budget_control.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault('release', {})[pod] = reason
    save(path, data)


def ssh(pod, command, data=None):
    result = subprocess.run([
        'ssh', '-i', str(Path.home()/'.ssh/runpod_deadline'), '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-p', str(pod['port']), 'root@'+pod['host'], command,
    ], input=data, text=True, capture_output=True, timeout=25,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])
    return result.stdout


def main():
    activated = False
    samples = []
    while True:
        config = json.loads((ROOT/'pods.json').read_text())
        pod = next(p for p in config['pods'] if p['id'] == NEW)
        if pod.get('paused') or pod.get('terminated'):
            return
        if not activated:
            ready = ssh(pod, 'test ! -f /workspace/logs/prepare.ready || echo READY').strip()
            if ready:
                job = dict(run=RUN, steps=127000, line='af60v8f57', stage='epoch57',
                           dataset='/root/kevin/datasets/af60v8f_ds_dagger2/lerobot/ds')
                ssh(pod, 'mkdir -p /workspace/queue; cat > /workspace/queue/10-'+RUN+'.job.json', json.dumps(job))
                # The ordinary supervisor now handles checkpoints and subsequent rounds.
                config = json.loads((ROOT/'pods.json').read_text())
                new = next(p for p in config['pods'] if p['id'] == NEW)
                new.update(role='pick', critical_only=True, max_round=3)
                save(ROOT/'pods.json', config)
                activated = True
                print('Unchanged B200 production R2 queued', flush=True)
        else:
            report = json.loads((ROOT/'status.json').read_text())
            states = {p['id']: p for p in report.get('pods', [])}
            b, h = states.get(NEW, {}), states.get(OLD, {})
            if b.get('failure_needs_review'):
                release(NEW, 'B200 production failed; original H100 run preserved')
                return
            now = report['time']
            point = (now, b.get('training_step', 0), h.get('training_step', 0))
            if not samples or point[0] > samples[-1][0]:
                samples.append(point)
            prior = next((s for s in reversed(samples) if now-s[0] >= 180 and s[1] > 100), None)
            if prior:
                elapsed = now-prior[0]
                bs, hs = (point[1]-prior[1])/elapsed, (point[2]-prior[2])/elapsed
                print(json.dumps(dict(b200_step=point[1], h100_step=point[2], b200_rate=bs, h100_rate=hs)), flush=True)
                if point[1] >= max(20000, point[2]) and bs > hs*1.1:
                    durable = ssh(pod, 'test ! -f /root/kevin/checkpoints/'+RUN+'/checkpoint-20000/model.safetensors.index.json || echo SAVED').strip()
                    if durable:
                        config = json.loads((ROOT/'pods.json').read_text())
                        next(p for p in config['pods'] if p['id'] == NEW).pop('benchmark_until', None)
                        save(ROOT/'pods.json', config)
                        release(OLD, 'B200 unchanged production caught up, checkpoint saved, and throughput improved')
                        print('H100 release requested; B200 carries production forward', flush=True)
                        return
                if len(samples) > 1 and now-samples[0][0] > 900 and bs > 0 and hs > 0:
                    b_remaining, h_remaining = (127000-point[1])/bs, (127000-point[2])/hs
                    if b_remaining >= h_remaining:
                        release(NEW, 'B200 would not finish earlier; original H100 run preserved')
                        return
        time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # No retry after SSH failure. Budget guard remains independent and bounded.
        print('Hardware handoff stopped: '+str(error), flush=True)
        release(NEW, 'Hardware handoff error; preserve original H100 training')
        raise
