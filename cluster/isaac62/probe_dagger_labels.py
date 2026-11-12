"""Bounded real student/expert DAgger label collection on Isaac Sim 6.2.

Uses the existing model interpreter in a separate process and the new simulator
interpreter for the actual collector. No study performance claim is made.
"""
import argparse
import contextlib
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
        raise SystemExit('Expose only the reviewed physical GPU 0')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['labels', 'rescue-broad', 'rescue-late'], default='labels',
                        help='Bounded schema gate or actual two-pass failure-conditioned rescue')
    parser.add_argument('--rescue-box', type=Path,
                        help='Existing expert-specific calibration; required for a rescue case')
    for key in ('checkpoint', 'model-python', 'sonic-pt', 'ref-motions-path', 'asset-cwd', 'output', 'expert-checkpoint', 'demo-root'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.case != 'labels' and args.rescue_box is None:
        parser.error('Rescue cases require an existing --rescue-box calibration')
    if args.rescue_box is not None:
        args.rescue_box = args.rescue_box.expanduser().resolve(strict=True)
    for key in ('checkpoint', 'sonic_pt', 'ref_motions_path', 'asset_cwd', 'expert_checkpoint', 'demo_root'):
        setattr(args, key, getattr(args, key).expanduser().resolve(strict=True))
    # A venv launcher is commonly a symlink to its base Python. Resolving that
    # symlink discards pyvenv.cfg discovery and launches the wrong environment.
    args.model_python = args.model_python.expanduser().absolute()
    if not args.model_python.is_file():
        parser.error('Model interpreter does not exist')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_output = output/'service'
    model_output.mkdir()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'Training/scripts/reinforcement_learning/rsl_rl'))
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    worker = client = None
    reserved = False
    # The own-motion fixture is sampled at 20Hz but the simulator steps at 50Hz.
    # Two hundred simulation steps stop before its reference grasp phase. Cover
    # the 12.6-second reference endpoint instead of silently skipping rescue.
    maximum_steps = 64 if args.case == 'labels' else 640
    receipt = dict(started=time.time(), completed=False, checkpoint=str(args.checkpoint),
                   model_python=str(args.model_python), simulator_python=sys.executable,
                   scientific_evaluation=False, maximum_steps=maximum_steps, case=args.case,
                   inherited_hs_settings={k: v for k, v in os.environ.items() if k.startswith('HS_')})
    if args.rescue_box is not None:
        receipt['rescue_box'] = dict(path=str(args.rescue_box),
                                   sha256=hashlib.sha256(args.rescue_box.read_bytes()).hexdigest())
    try:
        r = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                            '--mb', '24000', '--gpu', '0'], capture_output=True,
                           text=True, check=True, timeout=20)
        if r.stdout.strip() != '0':
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        with (output/'service.log').open('w') as stream:
            worker = subprocess.Popen([str(args.model_python), str(root/'cluster/isaac62/probe_gr00t_service.py'),
                                       '--checkpoint', str(args.checkpoint), '--output', str(model_output),
                                       '--physical-gpu', '0', '--worker'], stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+300
        while not (model_output/'ready.json').is_file():
            if worker.poll() is not None:
                raise RuntimeError('Model worker failed; inspect service.log')
            if time.monotonic() > deadline:
                raise TimeoutError('Model service readiness exceeded 300 seconds')
            time.sleep(.5)
        import h5py
        import numpy as np
        motion_ids = output/'motion_ids.txt'
        motion_ids.write_text('0\n')
        receipt['phases'] = []
        phases = [('student', '0'), ('expert', '1')] if args.case == 'labels' else [(args.case, '0')]
        for name, beta in phases:
            phase_output = output/name
            command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py'),
                       '--headless', '--device', 'cuda:0', '--num_envs', '1',
                       '--task', 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0',
                       '--waist-dof', '29', '--skip-start-frames', '0', '--seed', '17',
                       '--checkpoint-path', str(args.expert_checkpoint), '--sonic-pt', str(args.sonic_pt),
                       '--ref-motions-path', str(args.ref_motions_path), '--residual-transform', 'additive_free',
                       '--dagger-vla-checkpoint', str(args.checkpoint), '--gr00t-service', str(model_output/'ready.json'),
                       '--dagger-demo-root', str(args.demo_root), '--dagger-motion-ids', str(motion_ids),
                       '--dagger-beta', beta, '--dagger-chunk', '8', '--dagger-rollouts', '1',
                       '--num-samples', '1', '--rollout-length', str(maximum_steps), '--dagger-min-steps', '50',
                       '--output-directory', str(phase_output)]
            if args.case != 'labels':
                command += ['--dagger-rescue', '--dagger-rescue-box', 'file:'+str(args.rescue_box)]
                if args.case == 'rescue-late':
                    command += ['--dagger-rescue-window-s', '1']
            phase = dict(name=name, command=command, completed=False)
            receipt['phases'].append(phase)
            with (output/(name+'.log')).open('w') as stream:
                result = subprocess.run(command, cwd=args.asset_cwd, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=450)
            phase['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError('Actual DAgger '+name+' collector failed')
            paths = list(phase_output.rglob('*.hdf5'))
            if len(paths) != 1:
                raise RuntimeError('Expected exactly one retained DAgger recording')
            with h5py.File(paths[0]) as h:
                demo = h['data/demo_0']
                token = demo['obs/motion_token'][()]
                action = demo['actions'][()]
                fingers = demo['obs/robot0_right_finger_joint_pos'][()]
                mask = demo['obs/dagger_expert_mask'][()]
                frames = len(token)
                if not 1 <= frames <= maximum_steps:
                    raise RuntimeError('Unexpected recording length')
                if token.shape != (frames, 64) or action.shape != (frames, 65) or fingers.shape != (frames, 7):
                    raise RuntimeError('Unexpected recorded label shapes')
                if not all(np.isfinite(v).all() for v in (token, action, fingers)):
                    raise RuntimeError('Nonfinite expert labels')
                if args.case == 'labels':
                    if frames != maximum_steps or not np.all(mask == int(beta)):
                        raise RuntimeError('Expert-driver mask contradicts requested driver')
                else:
                    metadata = json.loads(h.attrs['metadata_json'])
                    rescue = metadata['rescue']
                    if rescue['pass'] != 'B' or rescue['passA_success'] is not False:
                        raise RuntimeError('Case did not exercise failure-conditioned expert rescue')
                    takeover = rescue['takeover_at']
                    lower, upper = rescue['takeover_bounds']
                    if not 0 <= lower <= takeover < upper <= rescue['grab_idx']:
                        raise RuntimeError('Takeover falls outside the saved uniform sampling bounds')
                    if args.case == 'rescue-broad' and (lower != 0 or upper != rescue['grab_idx']):
                        raise RuntimeError('Broad takeover bounds differ from the original rule')
                    if args.case == 'rescue-late' and rescue['window_s'] != 1:
                        raise RuntimeError('Late takeover window was not applied')
                    env_args = json.loads(h['data'].attrs['env_args'])
                    step_dt = float(env_args['step_dt'])
                    if not np.isfinite(step_dt) or step_dt <= 0:
                        raise RuntimeError('Recording lacks a valid simulation timestep')
                    if args.case == 'rescue-late' and lower != max(0, upper-round(1/step_dt)):
                        raise RuntimeError('Late sampling bounds do not match the actual one-second window')
                    expected_mask = (np.arange(frames) >= takeover).astype(mask.dtype)
                    if mask.shape != expected_mask.shape or not np.array_equal(mask, expected_mask):
                        raise RuntimeError('Recorded driver mask differs from the actual takeover schedule')
                    if not mask.any():
                        raise RuntimeError('No expert-driven frames recorded after takeover')
                    phase['rescue'] = rescue
                    phase['step_dt'] = step_dt
                if not np.allclose(token*16, np.round(token*16), atol=1e-7, rtol=0):
                    raise RuntimeError('Expert token labels do not lie on the FSQ grid')
                if not np.isin(action[:, 64], [-1, 1]).all():
                    raise RuntimeError('Nonbinary expert finger commands')
                closed = action[:, 64] < 0
                if not np.all(fingers[~closed] == 0):
                    raise RuntimeError('Open expert labels contain a nonzero finger target')
                if closed.any():
                    close_pose = fingers[closed][0]
                    if not np.any(close_pose != 0) or not np.all(fingers[closed] == close_pose):
                        raise RuntimeError('Closed expert labels differ from their fixed demonstration pose')
                phase.update(frames=len(token), expert_driver_frames=int(mask.sum()),
                             closed_label_frames=int((action[:, 64]<0).sum()),
                             hdf5=str(paths[0].relative_to(output)), completed=True)
        receipt.update(completed=True, finished=time.time())
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        if worker is not None:
            if worker.poll() is None and (model_output/'ready.json').is_file():
                with contextlib.suppress(Exception):
                    from vla_sonic.gr00t_service import PolicyServiceClient
                    client = PolicyServiceClient(model_output/'ready.json', args.checkpoint,
                                                 'cuda:0', 'unitree_g1_sonic', timeout_ms=10000)
                    client.shutdown_server()
            if client is not None:
                client.close()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'], check=True, timeout=20)
                receipt['reservation_released'] = True
        finally:
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
