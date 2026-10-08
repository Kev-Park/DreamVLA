"""Start pick evaluations/continuation when complete local checkpoints appear."""
from pathlib import Path
import subprocess
import time

from watch_dagger import checkpoint_complete

root = Path.home() / 'kevin'
jobs = {'af60v8f57_run01': 61000, 'af60v8f57_dagger1_run01': 97000,
        'af60v8f57_dagger2_run01': 127000}
while True:
    for run, steps in jobs.items():
        marker = root / 'dline/af60v8f57' / (run + '.bootstrap.done')
        if marker.exists():
            continue
        checkpoint = root / 'checkpoints' / run / f'checkpoint-{steps}'
        if not checkpoint_complete(checkpoint):
            continue
        subprocess.run(['python3', str(Path(__file__).with_name('bootstrap_epoch57.py')), run], check=True)
        marker.touch()
    time.sleep(30)
