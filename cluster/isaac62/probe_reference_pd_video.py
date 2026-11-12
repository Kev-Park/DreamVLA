"""Exercise the production reference-PD video and diagnostic overlay entrypoint.

This is a short technical rendering check. User review is required to assess
robot behavior and appearance; no task-performance estimate is produced.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
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
    for key in ('ref-motions-path', 'asset-cwd', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--mode', choices=['reference-pd', 'reference-playback'], default='reference-pd')
    parser.add_argument('--steps', type=int, default=32)
    args = parser.parse_args()
    if not 32 <= args.steps <= 768:
        parser.error('Bounded rendering requires 32..768 steps')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, scientific_evaluation=False,
                   reservation_released=False, mode=args.mode, steps=args.steps)
    stem = args.mode.replace('-', '_')+'_overlays'
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '16000', '--gpu', '0'], capture_output=True,
                                text=True, check=True, timeout=20)
        if result.stdout.strip() != '0':
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/play_sonic_adapter.py'),
                   '--headless', '--device', 'cuda:0', '--num_envs', '1',
                   '--task', 'Isaac-Motion-Tracking-Pick-HOI-v0',
                   '--'+args.mode, '--waist-dof', '29', '--force-motion-id', '0',
                   '--skip-start-frames', '0', '--seed', '17', '--video_length', str(args.steps),
                   '--ref-motions-path', str(args.ref_motions_path.expanduser().resolve(strict=True)),
                   '--overlay-ref', '--overlay-obj-candidates', '--overlay-contact',
                   '--name', str(output/(stem+'.mp4')), '--traj-dump', str(output/'trajectory.npz')]
        receipt['command'] = command
        with (output/'render.log').open('w') as stream:
            result = subprocess.run(command, cwd=args.asset_cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=max(180, args.steps))
        receipt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Production reference-PD video entrypoint failed')
        mode_marker = '[reference-pd]' if args.mode == 'reference-pd' else 'REFERENCE-PLAYBACK'
        for marker in (mode_marker, '[overlay-ref]', '[overlay-obj-candidates]'):
            if marker not in (output/'render.log').read_text():
                raise RuntimeError('Expected production path was not exercised: '+marker)
        ffprobe, ffmpeg = shutil.which('ffprobe'), shutil.which('ffmpeg')
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                 '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
                                 '-of', 'json', str(output/(stem+'.mp4'))],
                                capture_output=True, text=True, check=True, timeout=30)
        video = json.loads(result.stdout)['streams'][0]
        if int(video['nb_read_frames']) != args.steps:
            raise RuntimeError('Wrong reference-PD frame count')
        subprocess.run([ffmpeg, '-nostdin', '-loglevel', 'error', '-i', str(output/(stem+'.mp4')),
                        '-vf', 'select=eq(n\\,15)', '-frames:v', '1', str(output/'frame.png')],
                       check=True, timeout=30)
        import numpy as np
        from PIL import Image
        with np.load(output/'trajectory.npz', allow_pickle=False) as trace:
            if len(trace['motion_time']) != args.steps:
                raise RuntimeError('Wrong rendered trajectory length')
            for key in trace.files:
                if key not in ('joint_names', 'reference_joint_names') and not np.isfinite(trace[key]).all():
                    raise RuntimeError('Nonfinite rendered trajectory: '+key)
            if args.mode == 'reference-playback':
                for actual, reference in (('root_pos', 'ref_root_pos'), ('root_quat', 'ref_root_quat'),
                                          ('ref_order_joint_pos', 'ref_joint_pos')):
                    np.testing.assert_allclose(trace[actual], trace[reference], atol=1e-6, rtol=0)
            receipt['full_reference_clock_covered'] = bool(trace['motion_time'][-1] >= trace['motion_duration'][-1])
            receipt['motion_time_range'] = [float(trace['motion_time'][0]), float(trace['motion_time'][-1])]
            receipt['motion_duration'] = float(trace['motion_duration'][-1])
        pixels = np.asarray(Image.open(output/'frame.png').convert('RGB'))
        if pixels.max() == 0 or pixels.std() == 0:
            raise RuntimeError('Black/constant reference-PD frame')
        receipt.update(completed=True, video=video, frame_mean=float(pixels.mean()),
                       frame_std=float(pixels.std()), finished=time.time())
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
