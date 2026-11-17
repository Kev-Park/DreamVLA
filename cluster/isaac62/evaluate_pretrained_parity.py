"""Production expert evaluations, with original and migrated SDK/source pairs.

Launch only after agent review of account utilization and physical GPU contexts.
This changes neither policy outputs nor task success criteria.
"""
import hashlib
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
    output = base/'eval_videos/isaac62_expert_parity_20261116'
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    owner = str(os.getpid())
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    receipt = dict(started=time.time(), pid=os.getpid(), completed=False, runs=[])
    reserved = False

    def save():
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')

    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '9500', '--gpu', '0'], check=True,
                                capture_output=True, text=True, timeout=20)
        assert result.stdout.strip() == '0', 'Reviewed GPU unavailable'
        reserved = True
        for task, ckpt, pool, historical in [
            ('static', 'af60v8c_model_8998.pt', 'Holosoma_Pick_29_latband60max_eval', 88.35),
            ('walk', 'T20_max_v3_model_4500.pt', 'StitchMAX100v3_eval', 76.45),
        ]:
            for sdk in ['legacy', 'target']:
                name = task+'_'+sdk
                run_dir = output/name
                run_dir.mkdir()
                source = root if sdk == 'target' else base/('DreamVLA' if task == 'static' else 'wt/locomotion/DreamVLA')
                python = sys.executable if sdk == 'target' else str(Path.home()/'miniconda3/envs/dreamcontrol_51/bin/python')
                env = {k: v for k, v in os.environ.items() if not k.startswith('HS_')}
                env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cpu',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PXR_WORK_THREAD_LIMIT='2', PYTHONUNBUFFERED='1', HYBRID_RARM='0',
                           HS_REWORK='1', HS_REWORK_OBJ_PC='1', HS_REWORK_OBJ_GATE='1',
                           HS_REWORK_CONTACT_LEAD='2', HS_EVAL_CAMERAS='0',
                           HS_EVAL_DUMP_EPS=str(run_dir/'episodes.npz'))
                if task == 'walk':
                    env.update(HS_REWORK_CLOSE_ON_ARRIVAL='1', HS_REWORK_ARRIVE_FINGER_D='0',
                               HS_REWORK_ARRIVE_D='0.08', HS_REWORK_PALM_NORMAL='0',
                               HS_REWORK_PALM_REACH='0.07', HS_PALM_NORMAL_SIGN='+1')
                # Select the actual task/lab packages from the evaluated checkout.
                packages = [str(p) for p in (source/'Training/source').iterdir()
                            if p.is_dir() and (p/p.name/'__init__.py').is_file()]
                env['PYTHONPATH'] = os.pathsep.join(packages)
                checkpoint = base/'residuals'/ckpt
                refs = base/'ref_motions'/pool
                assert len(list(refs.glob('*.pkl'))) == 100
                command = [python, str(root/'cluster/isaac62/run_expert_entrypoint.py'),
                           '--entrypoint', str(source/'Training/scripts/reinforcement_learning/rsl_rl/eval_sonic_adapter.py'),
                           '--receipt', str(run_dir/'entrypoint.json'), '--', '--headless',
                           '--device', 'cuda:0', '--num_envs', '100', '--num_episodes', '500',
                           '--max_steps', '10000', '--seed', '0',
                           '--task', 'Isaac-Motion-Tracking-Pick-HOI-v0', '--waist-dof', '29',
                           '--sonic-pt', str(base/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'),
                           '--residual-transform', 'additive_free', '--residual-scale', '0.1',
                           '--ref-motions-path', str(refs), '--path', str(checkpoint),
                           '--kit_args=--/plugins/carb.tasking.plugin/threadCount=4']
                row = dict(name=name, started=time.time(), command=command,
                           source=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           reference_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(refs.glob('*.pkl'))},
                           hs_settings={k:v for k,v in env.items() if k.startswith('HS_')},
                           pythonpath=env['PYTHONPATH'], historical_held_success_pct=historical)
                receipt['runs'].append(row)
                save()
                with (run_dir/'eval.log').open('w') as log:
                    result = subprocess.run(command, cwd=base/'DreamVLA/Training', env=env,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=3600)
                row.update(returncode=result.returncode, finished=time.time())
                entry = json.loads((run_dir/'entrypoint.json').read_text()) if (run_dir/'entrypoint.json').exists() else {}
                row['entrypoint'] = entry
                row['metric_lines'] = [s for s in (run_dir/'eval.log').read_text().splitlines()
                                       if '[PHYS] held (lift>=5cm' in s or s.startswith('[MPJPE] ')]
                row['completed'] = result.returncode == 0 and entry.get('completed', False) and (run_dir/'episodes.npz').exists()
                save()
                # Keep the independent counterpart test even if one SDK fails.
        receipt['completed'] = all(r['completed'] for r in receipt['runs'])
    except BaseException:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        if reserved:
            subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'], check=True, timeout=20)
        receipt.update(finished=time.time(), reservation_released=reserved)
        save()


if __name__ == '__main__':
    main()
