"""Sequential physical-control review clips; requires prior agent resource review."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    base = Path.home()/'kevin'
    root = Path(__file__).resolve().parents[2]
    output = base/'eval_videos/isaac62_visual_review_20261116'
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PXR_WORK_THREAD_LIMIT='2', PYTHONUNBUFFERED='1')
    owner = str(os.getpid())
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    receipt = dict(started=time.time(), pid=os.getpid(), completed=False, runs=[],
                   scope='Physical-control visual review, not a success-rate evaluation',
                   source=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '10000', '--gpu', '0'], check=True, capture_output=True, text=True)
        assert result.stdout.strip() == '0'
        reserved = True
        for task, steps, checkpoint in [
            ('static', 500, base/'residuals/af60v7_model_4999.pt'),
            ('walk', 1200, base/'residuals/T20_max_v3_model_4500.pt'),
        ]:
            for mode in ['frozen', 'residual']:
                name = task+'_'+mode
                run_env = {k: v for k, v in env.items() if not k.startswith('HS_')}
                run_env.update(HS_REWORK='1', HS_REWORK_OBJ_PC='1', HS_REWORK_OBJ_GATE='1',
                               HS_REWORK_CONTACT_LEAD='2', HYBRID_RARM='0')
                if task == 'walk':
                    run_env.update(HS_REWORK_CLOSE_ON_ARRIVAL='1', HS_REWORK_ARRIVE_FINGER_D='0',
                                   HS_REWORK_ARRIVE_D='0.08', HS_REWORK_PALM_NORMAL='0',
                                   HS_REWORK_PALM_REACH='0.07', HS_PALM_NORMAL_SIGN='+1')
                refs = base/'validation'/('isaac62-'+('walk' if task == 'walk' else 'pick')+'-task-556890e-isolated/references')
                command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/play_sonic_adapter.py'),
                           '--headless', '--device', 'cuda:0', '--num_envs', '1', '--seed', '17',
                           '--task', 'Isaac-Motion-Tracking-Pick-'+('HOI' if task == 'walk' else 'BinaryFingers')+'-v0',
                           '--waist-dof', '29', '--skip-start-frames', '0', '--force-motion-id', '0',
                           '--keep-terms', '--no-overlay', '--video_length', str(steps),
                           '--name', str(output/(name+'_m0.mp4')), '--traj-dump', str(output/(name+'.npz')),
                           '--ref-motions-path', str(refs), '--residual-transform', 'additive_free',
                           '--sonic-pt', str(base/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000/model_step_100000.pt')]
                command += ['--zero-residual'] if mode == 'frozen' else ['--path', str(checkpoint)]
                row = dict(name=name, command=command, hs_settings={k:v for k,v in run_env.items() if k.startswith('HS_')},
                           started=time.time(), reference_files=[str(p.resolve()) for p in sorted(refs.glob('*.pkl'))])
                receipt['runs'].append(row)
                (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
                with (output/(name+'.log')).open('w') as log:
                    result = subprocess.run(command, cwd=base/'DreamVLA/Training', env=run_env,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                row.update(returncode=result.returncode, finished=time.time(),
                           video_exists=(output/(name+'_m0.mp4')).is_file(),
                           trace_exists=(output/(name+'.npz')).is_file())
                (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
                if result.returncode or not row['video_exists'] or not row['trace_exists']:
                    raise RuntimeError('Render failed: '+name+'; inspect its log')
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
