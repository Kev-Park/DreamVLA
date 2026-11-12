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
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, scientific_evaluation=False,
                   reservation_released=False, reference_pd=True, steps=32)
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
                   '--reference-pd', '--waist-dof', '29', '--force-motion-id', '0',
                   '--skip-start-frames', '0', '--seed', '17', '--video_length', '32',
                   '--ref-motions-path', str(args.ref_motions_path.expanduser().resolve(strict=True)),
                   '--overlay-ref', '--overlay-obj-candidates', '--overlay-contact',
                   '--name', str(output/'reference_pd_overlays.mp4')]
        receipt['command'] = command
        with (output/'render.log').open('w') as stream:
            result = subprocess.run(command, cwd=args.asset_cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        receipt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Production reference-PD video entrypoint failed')
        for marker in ('[reference-pd]', '[overlay-ref]', '[overlay-obj-candidates]'):
            if marker not in (output/'render.log').read_text():
                raise RuntimeError('Expected production path was not exercised: '+marker)
        ffprobe, ffmpeg = shutil.which('ffprobe'), shutil.which('ffmpeg')
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                 '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
                                 '-of', 'json', str(output/'reference_pd_overlays.mp4')],
                                capture_output=True, text=True, check=True, timeout=30)
        video = json.loads(result.stdout)['streams'][0]
        if int(video['nb_read_frames']) != 32:
            raise RuntimeError('Wrong reference-PD frame count')
        subprocess.run([ffmpeg, '-nostdin', '-loglevel', 'error', '-i', str(output/'reference_pd_overlays.mp4'),
                        '-vf', 'select=eq(n\\,15)', '-frames:v', '1', str(output/'frame.png')],
                       check=True, timeout=30)
        import numpy as np
        from PIL import Image
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
