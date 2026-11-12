"""Bounded real expert-collector and snapshot/replay migration checks.

Uses unchanged task recipes supplied by the launcher. A retained demo is not
guaranteed: a legitimate rejected episode is recorded as a rejection, not a pass
for successful-data writing. These are technical checks, not success estimates.
"""
import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    if os.name != 'posix' or importlib.metadata.version('isaacsim') != '6.2.0.0':
        raise SystemExit('Requires the isolated Linux Isaac Sim 6.2 environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise SystemExit('Requires one manually reviewed physical GPU 0')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sonic-pt', type=Path, required=True)
    parser.add_argument('--ref-motions-path', type=Path, required=True)
    parser.add_argument('--asset-cwd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-python', type=Path,
                        help='Existing legacy interpreter for a matched SDK baseline')
    parser.add_argument('--checks', nargs='+', choices=['determinism', 'restore', 'expert'],
                        default=['determinism', 'restore', 'expert'])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, scientific_evaluation=False,
                   reservation_released=False, phases=[])
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '16000', '--gpu', '0'], capture_output=True,
                                text=True, check=True, timeout=20)
        if result.stdout.strip() != '0':
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        runtime = args.runtime_python.expanduser().absolute() if args.runtime_python else Path(sys.executable)
        if not runtime.is_file():
            raise RuntimeError('Existing runtime interpreter not found')
        receipt['runtime_python'] = str(runtime)
        command = [str(runtime), str(root/'Training/scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py'),
                   '--headless', '--device', 'cuda:0', '--num_envs', '1',
                   '--checkpoint-path', str(args.checkpoint.expanduser().resolve(strict=True)),
                   '--sonic-pt', str(args.sonic_pt.expanduser().resolve(strict=True)),
                   '--ref-motions-path', str(args.ref_motions_path.expanduser().resolve(strict=True)),
                   '--task', 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0',
                   '--waist-dof', '29', '--skip-start-frames', '0',
                   '--seed', '17', '--motion-range', '0', '1', '--num-samples', '1',
                   '--rollout-length', '1200', '--residual-transform', 'additive_free']
        for name, extra, marker in (
                ('determinism', ['--dagger-selftest-determinism', '16'], '[selftest-determinism] PASS'),
                ('restore', ['--dagger-selftest-restore', '8', '16'], '[selftest-restore] PASS'),
                ('expert', [], '[INFO] Collection finished.')):
            if name not in args.checks:
                continue
            phase = dict(name=name, completed=False)
            receipt['phases'].append(phase)
            cmd = [*command, '--output-directory', str(output/name), *extra]
            phase['command'] = cmd
            with (output/(name+'.log')).open('w') as stream:
                result = subprocess.run(cmd, cwd=args.asset_cwd, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=300)
            phase['returncode'] = result.returncode
            log = (output/(name+'.log')).read_text()
            phase['marker_present'] = marker in log
            if result.returncode or marker not in log:
                phase['error'] = 'Actual collector gate failed; inspect '+name+'.log'
                # Independent checks must still run; preserve each failed gate.
                continue
            if name == 'expert':
                summary = re.search(r'Successes written: (\d+)/1 .*motions tried=(\d+)/1, '
                                    r'rejected_fall=(\d+), rejected_nograsp=(\d+), '
                                    r'rejected_topple=(\d+), errored=(\d+)', log)
                if summary is None:
                    raise RuntimeError('Missing exact collector episode accounting')
                counts = list(map(int, summary.groups()))
                if counts[1] != 1 or counts[-1] or counts[0]+sum(counts[2:5]) != 1:
                    raise RuntimeError('Invalid collector success/rejection/error accounting: '+str(counts))
                phase['episode_accounting'] = counts
            phase['completed'] = True
        import h5py
        import numpy as np
        demos = []
        for path in (output/'expert').rglob('*.hdf5'):
            datasets = {}
            with h5py.File(path) as stream:
                def inspect(name, value):
                    if not isinstance(value, h5py.Dataset):
                        return
                    datasets[name] = dict(shape=list(value.shape), dtype=str(value.dtype))
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value[()]).all():
                        raise RuntimeError('Nonfinite collected dataset: '+name)
                stream.visititems(inspect)
            if not datasets:
                raise RuntimeError('Empty collected HDF5')
            demos.append(dict(file=str(path.relative_to(output)), datasets=datasets))
        receipt.update(completed=all(p['completed'] for p in receipt['phases']),
                       demos=demos, retained_episode_count=len(demos),
                       successful_data_write_validated=bool(demos), finished=time.time())
        if not receipt['completed']:
            raise RuntimeError('One or more independent collector gates failed; inspect phase results')
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'], check=True, timeout=20)
                receipt['reservation_released'] = True
        finally:
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
