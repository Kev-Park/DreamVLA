"""Evaluate the established pick/walk students without changing their recipes.

One manually reviewed GPU hosts the model service and simulator sequentially.
Reuse the original four reference partitions, seeds and calibrated scorer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import re
import sys
import time
import traceback


def completed_shard_files(output, shard, ids):
    """Require a full summary and one finite closed trajectory per requested ID."""
    import numpy as np
    log = output/f'eval{shard}.log'
    if not log.exists():
        return None
    text = log.read_text(errors='replace')
    assert 'VLA + SONIC EVAL SUMMARY' in text and 'Traceback' not in text
    count = re.search(r'Completed episodes:\s*(\d+)', text)
    assert count and int(count[1]) == len(ids)
    files = sorted(output.glob(f'b{shard}_ep*_m*_traj.npz'))
    actual = [int(re.search(r'_m(\d+)_traj.npz$', p.name)[1]) for p in files]
    assert sorted(actual) == sorted(ids), 'Incomplete or duplicated reference coverage'
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for key in ['obj_pos', 'obj_quat', 'root_pos', 'root_quat',
                        'wrist', 'body_pos', 'ref_body_pos', 'ref_root_pos']:
                assert len(data[key]) > 0 and np.isfinite(data[key]).all(), (path, key)
    return [log, *files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path,
                        help='Reuse only independently audited complete shards from a closed run.')
    args = parser.parse_args()
    base = Path.home()/'kevin'
    root = Path(__file__).resolve().parents[2]
    os.sched_setaffinity(0, {0, 1, 2, 3})
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(pid=os.getpid(), started=time.time(), completed=False, runs=[])
    prior = None
    if args.resume_from:
        prior = json.loads((args.resume_from/'receipt.json').read_text())
        assert prior.get('reservation_released'), 'Prior run has not released its allocation'
        receipt['resume_from'] = str(args.resume_from)
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    reserved = False
    worker = None

    def save():
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))

    try:
        allocation = subprocess.check_output(
            [*slots, 'reserve', '--owner', owner, '--count', '1', '--mb', '26000',
             '--gpu', str(args.gpu)], text=True, timeout=20).strip()
        assert allocation == str(args.gpu)
        reserved = True
        for task, chain, checkpoint, pool, horizon in [
            ('pick', 'pick200_original_spares57_v2', '95899', 'Holosoma_Pick_29_latband60max_eval', 500),
            ('walk', 'walk200_v3_dagger57_20261013', '92967', 'StitchMAX100v3_eval', 1200),
        ]:
            output = args.output/task
            output.mkdir()
            source = base/'dline'/chain
            checkpoint = base/'checkpoints'/(chain+'_r2')/('checkpoint-'+checkpoint)
            refs = base/'ref_motions'/pool
            assert len(list(refs.glob('*.pkl'))) == 100
            env = {k:v for k,v in os.environ.items() if not k.startswith('HS_')}
            env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cpu',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PXR_WORK_THREAD_LIMIT='2', PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', HYBRID_RARM='0', HS_OBS_NOISE='0',
                       HS_REWORK='1', HS_REWORK_OBJ_PC='1', HS_REWORK_OBJ_GATE='1',
                       HS_REWORK_CONTACT_LEAD='2', HS_EVAL_NO_EE_TERM='1')
            if task == 'walk':
                env.update(HS_REWORK_CLOSE_ON_ARRIVAL='1', HS_REWORK_ARRIVE_FINGER_D='0',
                           HS_REWORK_ARRIVE_D='0.08', HS_REWORK_PALM_NORMAL='0',
                           HS_REWORK_PALM_REACH='0.07', HS_PALM_NORMAL_SIGN='+1')
            packages = [str(p) for p in (root/'Training/source').iterdir()
                        if p.is_dir() and (p/p.name/'__init__.py').is_file()]
            packages += [str(root/'Training/scripts/reinforcement_learning/rsl_rl')]
            env['PYTHONPATH'] = os.pathsep.join(packages)
            # These feature imports require no running simulator. Catch incomplete
            # deployments before loading a model or compiling renderer shaders.
            preflight = subprocess.run(
                [sys.executable, '-c',
                 'import isaac_utils.rotations; import vla_sonic.obs_to_policy; '
                 'import vla_sonic.gr00t_service; import pickle; from pathlib import Path; '
                 '[pickle.loads(p.read_bytes()) for p in Path('+repr(str(refs))+').glob("*.pkl")]'],
                env=env, capture_output=True, text=True, timeout=120)
            (output/'import_preflight.log').write_text(preflight.stdout+preflight.stderr)
            if preflight.returncode:
                raise RuntimeError('Offline student feature imports failed; inspect import_preflight.log')
            row = dict(task=task, checkpoint=str(checkpoint), started=time.time(), shards=[],
                       hs_settings={k:v for k,v in env.items() if k.startswith('HS_')})
            row['reference_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(refs.glob('*.pkl'))}
            receipt['runs'].append(row)
            save()
            model_env = env.copy()
            model_env['PYTHONPATH'] = str(base/'Isaac-GR00T')
            model_dir = output/'service'
            model_dir.mkdir()
            with (model_dir/'worker.log').open('w') as stream:
                worker = subprocess.Popen(
                    [str(base/'Isaac-GR00T/.venv/bin/python'),
                     str(root/'cluster/isaac62/probe_gr00t_service.py'), '--worker',
                     '--checkpoint', str(checkpoint), '--output', str(model_dir),
                     '--physical-gpu', str(args.gpu)], env=model_env,
                    stdout=stream, stderr=subprocess.STDOUT)
                row['worker_pid'] = worker.pid
                save()
                deadline = time.monotonic()+900
                while not (model_dir/'ready.json').exists():
                    if worker.poll() is not None:
                        raise RuntimeError('Model service exited; inspect its worker.log')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Model service readiness exceeded 900 seconds')
                    time.sleep(1)
                row['checkpoint_identity'] = json.loads((model_dir/'ready.json').read_text())['checkpoint']
                if prior:
                    old = next((r for r in prior['runs'] if r['task'] == task), None)
                    if old:
                        for key in ['reference_sha256', 'hs_settings']:
                            assert old[key] == row[key], 'Resume recipe differs: '+key
                        assert old['checkpoint_identity']['files_sha256'] == row['checkpoint_identity']['files_sha256']
                save()
                for shard in range(4):
                    ids = source/f'eval_ids{shard}.txt'
                    if prior:
                        files = completed_shard_files(args.resume_from/task, shard,
                                                      list(map(int, ids.read_text().split())))
                        if files:
                            for file in files:
                                (output/file.name).symlink_to(file.resolve(strict=True))
                            row['shards'].append(dict(shard=shard, completed=True,
                                reused_from=str(args.resume_from/task),
                                proof='Complete summary, exact reference coverage, finite original trajectories',
                                sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
                            save()
                            continue
                    command = [sys.executable, str(root/'cluster/isaac62/run_expert_entrypoint.py'),
                               '--entrypoint', str(root/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'),
                               '--receipt', str(output/f'entrypoint{shard}.json'), '--',
                               '--headless', '--sonic-pt', str(base/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'),
                               '--ref-motions-path', str(refs), '--vla-checkpoint', str(checkpoint),
                               '--gr00t-service', str(model_dir/'ready.json'), '--waist-dof', '29',
                               '--motions-from', str(ids), '--num-episodes', '25', '--seed', str(100+shard),
                               '--traj-dump', str(output/f'b{shard}'),
                               '--max-steps-per-episode', str(horizon),
                               '--kit_args=--/plugins/carb.tasking.plugin/threadCount=4']
                    item = dict(shard=shard, command=command, started=time.time())
                    row['shards'].append(item)
                    save()
                    with (output/f'eval{shard}.log').open('w') as log:
                        result = subprocess.run(command, cwd=base/'DreamVLA/Training', env=env,
                                                stdout=log, stderr=subprocess.STDOUT, timeout=21600)
                    entry_path = output/f'entrypoint{shard}.json'
                    entry = json.loads(entry_path.read_text()) if entry_path.exists() else {}
                    text = (output/f'eval{shard}.log').read_text(errors='replace')
                    item.update(returncode=result.returncode, finished=time.time(), entrypoint=entry)
                    save()
                    if result.returncode or not entry.get('completed') or 'VLA + SONIC EVAL SUMMARY' not in text:
                        raise RuntimeError(f'{task} shard {shard} failed: {result.returncode}')
            worker.terminate()
            worker.wait(timeout=30)
            worker = None
            # The common scorer uses the unchanged task-specific calibrated box.
            result = subprocess.run([sys.executable, str(root/'cluster/pick_budget_score.py'),
                                     str(output), str(source/'box.json')], env=env,
                                    capture_output=True, text=True, timeout=120)
            (output/'score.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError(f'{task} scoring failed')
            report = json.loads((output/'results.json').read_text())
            # pick_budget_score shares the algorithm, but its pool label is static.
            report['pool'] = pool
            holds = []
            for shard in range(4):
                match = re.search(r'\[PHYS\] held \(lift>=5cm[^\n]*?:\s*(\d+)\s*/\s*(\d+)',
                                  (output/f'eval{shard}.log').read_text(errors='replace'))
                assert match, f'Missing physical hold summary for shard {shard}'
                holds.append(tuple(map(int, match.groups())))
            assert sum(n for _,n in holds) == 100
            report['physical_held5_successes'] = sum(k for k,_ in holds)
            (output/'results.json').write_text(json.dumps(report, indent=2))
            row.update(completed=True, finished=time.time(), results=report)
            save()
        receipt['completed'] = True
    except BaseException:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
        if reserved:
            subprocess.run([*slots, 'release', '--owner', owner, '--gpu', str(args.gpu)],
                           check=True, timeout=20)
        receipt.update(finished=time.time(), reservation_released=reserved)
        save()


if __name__ == '__main__':
    main()
