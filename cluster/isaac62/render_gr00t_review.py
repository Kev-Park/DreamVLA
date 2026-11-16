"""Physical walk+pick render using an already authenticated GR00T service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', type=Path, required=True)
    args = parser.parse_args()
    base = Path.home()/'kevin'
    root = Path(__file__).resolve().parents[2]
    output = base/'eval_videos/isaac62_visual_gr00t_review_20261116'
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PXR_WORK_THREAD_LIMIT='2', PYTHONUNBUFFERED='1',
               HS_REWORK='1', HS_REWORK_OBJ_PC='1', HS_REWORK_OBJ_GATE='1',
               HS_REWORK_CONTACT_LEAD='2', HYBRID_RARM='0',
               HS_REWORK_CLOSE_ON_ARRIVAL='1', HS_REWORK_ARRIVE_FINGER_D='0',
               HS_REWORK_ARRIVE_D='0.08', HS_REWORK_PALM_NORMAL='0',
               HS_REWORK_PALM_REACH='0.07', HS_PALM_NORMAL_SIGN='+1')
    owner = str(os.getpid())
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    receipt = dict(started=time.time(), pid=os.getpid(), completed=False,
                   scope='Trained walk DAgger-2 visual control review; no success-rate estimate')
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '10000', '--gpu', '0'], check=True, capture_output=True, text=True)
        assert result.stdout.strip() == '0'
        reserved = True
        command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/play_vla_sonic.py'),
                   '--headless', '--device', 'cuda:0', '--num-envs', '1', '--seed', '17',
                   '--task', 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0',
                   '--waist-dof', '29', '--skip-start-frames', '0', '--motion-ids', '0',
                   '--num-episodes', '1', '--max-steps-per-episode', '1200',
                   '--record-video', str(output/'walk_gr00t'), '--traj-dump', str(output/'trace'),
                   '--language', 'Walk to the object and pick it up.',
                   '--ref-motions-path', str(base/'validation/isaac62-walk-task-556890e-isolated/references'),
                   '--vla-checkpoint', str(base/'checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'),
                   '--gr00t-service', str(args.service),
                   '--sonic-pt', str(base/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000/model_step_100000.pt')]
        receipt.update(command=command, hs_settings={k:v for k,v in env.items() if k.startswith('HS_')})
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
        with (output/'render.log').open('w') as log:
            result = subprocess.run(command, cwd=base/'DreamVLA/Training', env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        receipt.update(returncode=result.returncode,
                       videos=[p.name for p in output.glob('*.mp4')],
                       traces=[p.name for p in output.glob('*.npz')])
        if result.returncode or not receipt['videos'] or not receipt['traces']:
            raise RuntimeError('GR00T render failed; inspect render.log')
        receipt['completed'] = True
    except BaseException:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        if reserved:
            subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'], check=True)
        receipt.update(finished=time.time(), reservation_released=reserved)
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
