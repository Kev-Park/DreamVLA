"""Repeat the frozen Pick R2/R3 evaluation with paired, predeclared RNG seeds.

Orchestration only: scientific evaluation/scoring code stays in its pinned checkout.
No training, collection, checkpoint changes, automatic retries, or success filtering.
"""
import concurrent.futures
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

from pick_original_bluesclues_queue import capacity, revalidated_capacity

TAG = 'pick200_original_spares57_v2'
PIN = '197a3c7f9bce10604f4c8100c0d72d7549978377'
POOL = 'Holosoma_Pick_29_latband60max_eval'
STEPS = {2: 95899, 3: 123802}
ROOT = Path.home() / 'kevin'


def write(path, value):
    temporary = path.with_suffix(f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def plan(study):
    return [dict(round=r, seed_base=seed, shard=k, seed=seed+k,
                 ids=list(range(k, 100, 4)), run=f'{study}_r{r}_s{seed}')
            for seed in [200, 300, 400] for r in [2, 3] for k in range(4)]


def validate_report(report):
    assert report['pool'] == POOL and report['episodes'] == 100
    assert sorted(r['motion_id'] for r in report['per_episode']) == list(range(100))
    assert report['successes'] == sum(r['success'] for r in report['per_episode'])
    for metric in ['world_mpjpe_cm', 'root_aligned_mpjpe_cm']:
        assert math.isfinite(report[metric]) and report[metric] >= 0
        assert all(math.isfinite(r[metric]) and r[metric] >= 0 for r in report['per_episode'])
        assert abs(sum(r[metric] for r in report['per_episode'])/100-report[metric]) < 1e-8


def reserve_devices(request):
    """Use the existing shared registry; touch only this controller's leases."""
    import fcntl
    gpus = request['gpus']
    masks = request['cpu_masks']
    assert len(gpus) == len(masks) == 4 and len(set(gpus)) == 4
    assert all(len(mask) == 4 for mask in masks)
    assert len({c for mask in masks for c in mask}) == 16
    assert all(c in range(os.cpu_count()) for mask in masks for c in mask)
    before = capacity()
    directory = ROOT / 'dline'
    with (directory/'gpu_slots.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = revalidated_capacity(before, capacity())
        assert current is not None, 'Incomplete observation; no allocation made'
        registry = directory/'gpu_slots.json'
        entries = json.loads(registry.read_text()) if registry.exists() else []
        live = [e for e in entries if Path('/proc', str(e['owner'])).exists()]
        occupied = set(current['occupied']) | {e['gpu'] for e in live}
        assert not (set(gpus) & occupied) and set(gpus) <= set(current['free'])
        assert len(occupied | set(gpus) | {0}) <= 5
        assert current['cpu_fraction'] + 16/os.cpu_count() < .198
        additions = [dict(owner=os.getpid(), gpu=gpu, mb=45000,
                          study=request['study']) for gpu in gpus]
        write(registry, entries + additions)
    return current


def release_devices(gpus):
    import fcntl
    directory = ROOT/'dline'
    with (directory/'gpu_slots.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory/'gpu_slots.json'
        entries = json.loads(path.read_text())
        write(path, [e for e in entries
                     if not (e['owner'] == os.getpid() and e['gpu'] in gpus)])


def verify_inputs(request):
    runtime = ROOT/'wt/pick200-original-spares'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime,
                                   text=True).strip() == PIN
    for name in request['scientific_files']:
        expected = subprocess.check_output(['git', 'show', f'{PIN}:{name}'], cwd=runtime)
        assert (runtime/name).read_bytes() == expected, name
    baseline = ROOT/'dline'/TAG
    original = json.loads((baseline/'chain.request.final.json').read_text())
    files = [Path(p) for p in glob.glob(str(ROOT/'ref_motions'/POOL/'*.pkl'))]
    expected = original['eval_reference_manifest']['files']
    assert len(files) == len(expected) == 100
    assert [p.name for p in files] == [r['name'] for r in expected]
    for p, row in zip(files, expected):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    env = dict(original['environment'], HS_EVAL_NO_EE_TERM='1')
    assert env == request['eval_environment'] and env['HS_EVAL_NO_EE_TERM'] == '1'
    for r, step in STEPS.items():
        ck = ROOT/'checkpoints'/f'{TAG}_r{r}'/f'checkpoint-{step}'
        assert json.loads((ck/'trainer_state.json').read_text())['global_step'] == step
        index = json.loads((ck/'model.safetensors.index.json').read_text())
        assert all((ck/n).stat().st_size > 0 for n in set(index['weight_map'].values()))
    return runtime, original['box']


def run(request_file):
    import fcntl
    request = json.loads(request_file.read_text())
    work = request_file.parent
    lock = (work/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert request['seed_bases'] == [200, 300, 400]
    assert not list(work.glob('*.process.json')), 'Prior execution exists; inspect before resuming'
    runtime, box = verify_inputs(request)
    tasks = plan(request['study'])
    for run_name in sorted({t['run'] for t in tasks}):
        output = ROOT/'eval_videos'/run_name
        assert not output.exists(), f'Output exists: {output}'
    allocated = False
    started = time.time()
    write(work/'controller.json', dict(pid=os.getpid(), started=started, phase='preflight'))
    try:
        snapshot = reserve_devices(request)
        allocated = True
        write(work/'allocation.json', dict(time=time.time(), gpus=request['gpus'],
            cpu_masks=request['cpu_masks'], capacity=snapshot, workers_per_gpu=2))
        for run_name in sorted({t['run'] for t in tasks}):
            (ROOT/'eval_videos'/run_name).mkdir()
        write(work/'box.json', box)
        write(work/'plan.json', tasks)
        pending = queue.Queue()
        for task in tasks:
            pending.put(task)
        completed = []
        failures = []
        mutex = threading.Lock()
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
        for key in list(env):
            if key.startswith('HS_'):
                del env[key]
        env.update(request['eval_environment'])
        env['PYTHONPATH'] = str(runtime/'Training/scripts/reinforcement_learning/rsl_rl')

        def worker(lane):
            gpu = request['gpus'][lane//2]
            mask = request['cpu_masks'][lane//2]
            while True:
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    return
                label = f"{task['run']}_shard{task['shard']}"
                ids = work/f'{label}.ids.txt'
                ids.write_text(' '.join(map(str, task['ids']))+'\n')
                output = ROOT/'eval_videos'/task['run']
                checkpoint = ROOT/'checkpoints'/f"{TAG}_r{task['round']}"/f"checkpoint-{STEPS[task['round']]}"
                command = ['taskset', '-c', ','.join(map(str, mask)), sys.executable,
                    str(runtime/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'),
                    '--headless', '--sonic-pt', str(ROOT/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'),
                    '--ref-motions-path', POOL, '--vla-checkpoint', str(checkpoint),
                    '--waist-dof', '29', '--motions-from', str(ids), '--num-episodes', '25',
                    '--seed', str(task['seed']), '--traj-dump', str(output/f"b{task['shard']}")]
                try:
                    with (work/f'{label}.log').open('x') as log:
                        child = subprocess.Popen(command, cwd=runtime/'Training',
                            env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdout=log, stderr=subprocess.STDOUT)
                        receipt = dict(task, pid=child.pid, started=time.time(), gpu=gpu,
                                       cpu_mask=mask, command=command)
                        write(work/f'{label}.process.json', receipt)
                        code = child.wait()
                    write(work/f'{label}.exit.json', dict(code=code, time=time.time()))
                    assert code == 0, (label, code)
                    paths = list(output.glob(f"b{task['shard']}_*_traj.npz"))
                    assert sorted(int(re.search(r'_m(\d+)_traj.npz$', p.name)[1])
                                  for p in paths) == task['ids']
                    assert all(p.stat().st_size > 0 for p in paths)
                    with mutex:
                        completed.append(label)
                except Exception as error:
                    with mutex:
                        failures.append(dict(label=label, error=repr(error)))
                    return  # Preserve outputs; no automatic retry or termination.
                finally:
                    with mutex:
                        write(work/'state.json', dict(time=time.time(), phase='evaluating',
                            completed=list(completed), failures=list(failures), planned=24))

        write(work/'controller.json', dict(pid=os.getpid(), started=started, phase='evaluating'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(worker, range(8)))
        assert not failures and len(completed) == 24, failures
        release_devices(request['gpus'])
        allocated = False
        reports = []
        for run_name in sorted({t['run'] for t in tasks}):
            output = ROOT/'eval_videos'/run_name
            command = [sys.executable, str(runtime/'cluster/pick_budget_score.py'),
                       str(output), str(work/'box.json')]
            with (work/f'{run_name}.score.log').open('x') as log:
                subprocess.run(command, cwd=runtime, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            report = json.loads((output/'results.json').read_text())
            validate_report(report)
            reports.append(dict(run=run_name, report=report))
        write(work/'results.json', dict(time=time.time(), reports=reports,
            scope='Three additional evaluation RNG replicates per fixed checkpoint; no training-seed replication'))
        write(work/'controller.json', dict(pid=os.getpid(), started=started,
              phase='complete', finished=time.time(), completed_shards=24, episodes=600))
    except BaseException as error:
        write(work/'error.json', dict(time=time.time(), error=repr(error),
              experiment_jobs_restarted=False))
        raise
    finally:
        if allocated:
            release_devices(request['gpus'])


if __name__ == '__main__':
    run(Path(sys.argv[1]))
