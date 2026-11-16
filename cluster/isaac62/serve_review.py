"""Bounded, separately hosted GR00T service for simulator review renders."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(exist_ok=False, parents=True)
    os.sched_setaffinity(0, {0, 1})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    owner = str(os.getpid())
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1', '--mb', '14000',
                             '--gpu', str(args.gpu)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == str(args.gpu)
    row = dict(pid=os.getpid(), started=time.time(), gpu=args.gpu, checkpoint=str(args.checkpoint))
    worker = None
    try:
        with (args.output/'worker.log').open('w') as log:
            worker = subprocess.Popen([sys.executable, str(root/'cluster/isaac62/probe_gr00t_service.py'),
                                       '--worker', '--checkpoint', str(args.checkpoint), '--output', str(args.output),
                                       '--physical-gpu', str(args.gpu)], env=env, stdout=log, stderr=subprocess.STDOUT)
            row['worker_pid'] = worker.pid
            (args.output/'owner.json').write_text(json.dumps(row, indent=2))
            row['returncode'] = worker.wait(timeout=3600)
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
        subprocess.run([*slots, 'release', '--owner', owner, '--gpu', str(args.gpu)], check=True)
        row.update(finished=time.time(), reservation_released=True)
        (args.output/'owner.json').write_text(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
