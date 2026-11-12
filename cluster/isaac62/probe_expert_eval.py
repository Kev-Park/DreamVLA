"""One completed episode through the real nominal/residual expert evaluator.

This is a migration smoke gate, not a policy performance estimate. Run only on a
manually reviewed Linux GPU in the isolated simulator environment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    if os.name != 'posix' or importlib.metadata.version('isaacsim') != '6.2.0.0':
        raise SystemExit('Requires the isolated Linux Isaac Sim 6.2 environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise SystemExit('This bounded gate requires the reviewed physical GPU 0 only')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--ref-motions-path', type=Path, required=True)
    parser.add_argument('--sonic-pt', type=Path, required=True)
    parser.add_argument('--asset-cwd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--residual-transform', default='additive_free',
                        choices=['additive', 'additive_free', 'multiplicative',
                                 'multiplicative_free', 'unclamped', 'direct'])
    args = parser.parse_args()
    for name in ('checkpoint', 'ref_motions_path', 'sonic_pt', 'asset_cwd'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve(strict=True))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, scientific_evaluation=False,
                   scope='At least one completed episode; real evaluator and body MPJPE',
                   checkpoint=str(args.checkpoint) if args.checkpoint else None,
                   residual_transform=args.residual_transform, reservation_released=False)
    reserved = False
    try:
        r = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                            '--mb', '16000', '--gpu', '0'], capture_output=True,
                           text=True, check=True, timeout=20)
        if r.stdout.strip() != '0':
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        if args.checkpoint:
            with args.checkpoint.open('rb') as stream:
                receipt['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        env = os.environ.copy()
        env.update(HS_EVAL_CAMERAS='0', HS_EVAL_DUMP_EPS=str(output/'episodes.npz'))
        command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/eval_sonic_adapter.py'),
                   '--headless', '--device', 'cuda:0', '--num_envs', '2',
                   '--num_episodes', '1', '--max_steps', '800', '--seed', '17',
                   '--motion-id', '0', '--task', 'Isaac-Motion-Tracking-Pick-BinaryFingers-v0',
                   '--waist-dof', '29', '--skip-start-frames', '0',
                   '--sonic-pt', str(args.sonic_pt),
                   '--ref-motions-path', str(args.ref_motions_path),
                   '--residual-transform', args.residual_transform]
        command += ['--path', str(args.checkpoint)] if args.checkpoint else ['--zero-residual']
        receipt.update(command=command, inherited_hs_settings={
            key: value for key, value in env.items() if key.startswith('HS_')})
        log = output/'eval.log'
        with log.open('w') as stream:
            r = subprocess.run(command, cwd=args.asset_cwd, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=900)
        receipt['returncode'] = r.returncode
        if r.returncode:
            raise RuntimeError('Actual evaluator failed; inspect eval.log')
        lines = [line[len('[MPJPE] '):] for line in log.read_text().splitlines()
                 if line.startswith('[MPJPE] ')]
        if len(lines) != 1:
            raise RuntimeError('Expected one actual evaluator MPJPE result')
        metrics = json.loads(lines[0])
        if metrics['frames'] < 1 or metrics['completed_episodes'] < 1:
            raise RuntimeError('No completed episode contributed to MPJPE')
        for key in ('world_cm', 'root_relative_cm'):
            if not math.isfinite(metrics[key]) or metrics[key] < 0:
                raise RuntimeError('Invalid body error: '+key)
        if not (output/'episodes.npz').is_file():
            raise RuntimeError('Actual evaluator did not save episode records')
        receipt.update(completed=True, mpjpe=metrics, finished=time.time())
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'],
                               check=True, timeout=20)
                receipt['reservation_released'] = True
        finally:
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
