"""Check actual recorded-token/finger replay without changing the accepted decoder.

Exact requested-action equality is tested. This does not claim physical replay
equality: the existing replay decoder and residual expert use different FSQ paths.
"""
import argparse
import hashlib
import importlib.metadata
import json
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
        raise SystemExit('Requires one manually reviewed physical GPU 0')
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('hdf5', 'sonic-pt', 'ref-motions-path', 'asset-cwd', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    reserved = False
    receipt = dict(started=time.time(), completed=False, scientific_evaluation=False,
                   reservation_released=False, physical_replay_equality_validated=False)
    try:
        import h5py
        import numpy as np
        source = args.hdf5.expanduser().resolve(strict=True)
        with h5py.File(source) as h:
            demo = h['data/demo_0']
            expected = np.column_stack((demo['obs/motion_token'][()], demo['actions'][()][:, 64])).astype(np.float32)
            metadata = json.loads(h.attrs['metadata_json'])
        if len(expected) != 64 or expected.shape[1] != 65:
            raise RuntimeError('Requires the actual complete 64-frame expert-driver smoke recording')
        if metadata.get('skip_start_frames') is None:
            raise RuntimeError('Cannot infer recorded reference start')
        # Collector reserves the preceding ten frames for decoder-history warmup.
        start = int(metadata['skip_start_frames'])+10
        with source.open('rb') as stream:
            receipt['source_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '16000', '--gpu', '0'], capture_output=True,
                                text=True, check=True, timeout=20)
        if result.stdout.strip() != '0':
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'),
                   '--headless', '--device', 'cuda:0', '--num-envs', '1', '--num-episodes', '1',
                   '--max-steps-per-episode', '64', '--seed', '17', '--waist-dof', '29',
                   '--skip-start-frames', str(start), '--task', 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0',
                   '--sonic-pt', str(args.sonic_pt), '--ref-motions-path', str(args.ref_motions_path),
                   '--replay-hdf5', str(source), '--traj-dump', str(output/'trace')]
        receipt['command'] = command
        with (output/'replay.log').open('w') as stream:
            result = subprocess.run(command, cwd=args.asset_cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        receipt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Actual recorded-action evaluator failed')
        trace_path = output/f"trace_ep0_m{int(metadata['motion_id'])}_traj.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            if not np.array_equal(trace['requested_action'], expected):
                raise RuntimeError('Replay requested actions differ from recorded tokens/finger commands')
            for key in trace.files:
                if key != 'body_names' and not np.isfinite(trace[key]).all():
                    raise RuntimeError('Nonfinite replay trace: '+key)
            receipt.update(completed=True, frames=len(trace['root_pos']),
                           requested_actions_exact=True, recorded_start_frame=start,
                           closed_command_frames=int((expected[:, 64]<0).sum()), finished=time.time())
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
