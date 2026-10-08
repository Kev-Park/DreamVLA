"""One-GPU, file-backed worker queue. Jobs are data; all code comes from git."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/workspace/queue')
ROOT.mkdir(exist_ok=True)


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def complete(checkpoint):
    index = checkpoint / 'model.safetensors.index.json'
    if not index.exists():
        return False
    shards = set(json.loads(index.read_text())['weight_map'].values())
    return all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in shards)


while not (ROOT / 'stop').exists():
    pending = sorted(ROOT.glob('*.job.json'))
    if not pending:
        time.sleep(10)
        continue
    path = pending[0]
    active = path.with_suffix('.active')
    path.rename(active)
    job = json.loads(active.read_text())
    name, steps = job['run'], int(job['steps'])
    checkpoint = Path.home() / 'kevin/checkpoints' / name / f'checkpoint-{steps}'
    log = Path('/workspace/logs') / f'{name}.log'
    status = dict(job, started=time.time(), state='running')
    write(ROOT / 'current.json', status)
    with log.open('ab') as output:
        child = subprocess.Popen(['bash', str(Path(__file__).with_name('runpod_finetune.sh')), job['dataset'], name, str(steps)], stdout=output, stderr=subprocess.STDOUT)
        status['pid'] = child.pid
        write(ROOT / 'current.json', status)
        rc = child.wait()
    status.update(finished=time.time(), exit_code=rc, state='done' if rc == 0 and complete(checkpoint) else 'failed')
    write(ROOT / f'{name}.result.json', status)
    write(ROOT / 'current.json', status)
    active.rename(active.with_suffix('.finished'))
    if status['state'] == 'failed':
        # Preserve the failure and stop; never silently retry a broken recipe.
        raise SystemExit(f'{name} failed: inspect {log}')
