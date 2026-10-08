"""One-GPU, file-backed worker queue. Jobs are data; all code comes from git."""
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl

ROOT = Path('/workspace/queue')
ROOT.mkdir(exist_ok=True)
lock = (ROOT / 'worker.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


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
    batch = int(job.get('global_batch_size', 16))
    if batch not in (16, 32):
        raise ValueError(f'Unsupported experiment batch size: {batch}')
    checkpoint = Path.home() / 'kevin/checkpoints' / name / f'checkpoint-{steps}'
    log = Path('/workspace/logs') / f'{name}.log'
    status = dict(job, started=time.time(), state='running')
    write(ROOT / 'current.json', status)
    with log.open('ab') as output:
        child = subprocess.Popen(['bash', str(Path(__file__).with_name('runpod_finetune.sh')), job['dataset'], name, str(steps), str(batch)], stdout=output, stderr=subprocess.STDOUT)
        status['pid'] = child.pid
        write(ROOT / 'current.json', status)
        rc = child.wait()
    status.update(finished=time.time(), exit_code=rc, state='done' if rc == 0 and complete(checkpoint) else 'failed')
    if status['state'] == 'failed':
        status['error_log_tail'] = log.read_bytes()[-24000:].decode('utf-8', errors='replace')
    write(ROOT / f'{name}.result.json', status)
    write(ROOT / 'current.json', status)
    active.rename(active.with_suffix('.finished'))
    if status['state'] == 'failed':
        # Preserve the failure and stop; never silently retry a broken recipe.
        raise SystemExit(f'{name} failed: inspect {log}')
