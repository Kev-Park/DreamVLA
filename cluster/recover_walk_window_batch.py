"""Operator recovery of the reviewed walking-window collection after reboot.

Completed attempts and original logs are immutable. Only unfinished references
are reissued; the interrupted attempts and RNG restart are recorded explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from gr00t_budget_chain import Chain, HERE, ROOT, atomic, read, verify_collection_logs
from resume_walk_after_reboot import inspect_recordings, progress
from walk_late_takeover_collect import audit_batch
from watch_t20max_v3 import environment

TAG = 'walk200_v3_histwindow65_20261029'
SUFFIX = 'reboot_20261031'


def identity(pid, name):
    try:
        cmd = Path('/proc', str(pid), 'cmdline').read_bytes().decode(errors='replace')
        return name in cmd and TAG in cmd
    except OSError:
        return False


def build_plan(batch):
    work = ROOT / 'dline' / TAG
    name = f'batch_{batch:03d}'
    assert not (work / (name + '.complete.json')).exists()
    expected = list(range(batch * 10, batch * 10 + 10))
    log = work / (name + '.log')
    proc = read(work / (name + '.process.json'))
    assert not identity(proc['pid'], 'collect_sonic_adapter.py')
    boot_time = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    assert proc['started'] < boot_time, 'Only a proven pre-reboot worker may be recovered'
    command = list(map(str, proc['command']))
    for flag, value in [('--seed', '9'), ('--dagger-rescue-window-s', '6.5'),
                        ('--dagger-chunk', '8'), ('--dagger-rollouts', '1'),
                        ('--ref-motions-path', 'StitchMAX200v3')]:
        assert command[command.index(flag) + 1] == value, flag
    state = progress([log.read_text()], expected)
    recordings = inspect_recordings(work / 'raw' / name, state['written_ids'])
    plan = dict(state, tag=TAG, batch=name, expected_ids=expected,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                boot_time=boot_time, original_command=command,
                original_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                recordings=recordings, checked_at=time.time(),
                rng_note='Recovery worker restarts seed 9; only unfinished references are issued. Completed demonstrations are reused unchanged.')
    path = work / (name + '.' + SUFFIX + '.plan.json')
    if path.exists():
        old = read(path)
        assert all(old[k] == v for k, v in plan.items() if k != 'checked_at')
        return old
    atomic(path, plan)
    return plan


def run(plan, gpu):
    import fcntl
    import subprocess
    from account_cpu_usage import sample_account_cpu

    chain = Chain(200, tag=TAG)
    work = chain.work
    name = plan['batch']
    stage = name + '_' + SUFFIX
    lock = (work / (name + '.recovery.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id']
    assert not (work / (stage + '.workflow.json')).exists(), 'Existing recovery needs reconciliation'
    assert hashlib.sha256((work / (name + '.log')).read_bytes()).hexdigest() == plan['original_log_sha256']
    assert not (work / (name + '.complete.json')).exists()
    cpu = sample_account_cpu(3)
    assert cpu['cpu_measurement_complete']
    assert cpu['account_cpu_pct'] + len(os.sched_getaffinity(0)) / cpu['logical_cpus'] * 100 <= 20
    state = {'status': 'starting', 'pid': os.getpid(), 'started': time.time(),
             'tag': TAG, 'batch': name, 'stage': stage, 'gpu': gpu,
             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'account_cpu_before': cpu,
             'reused_ids': plan['completed_ids'], 'remaining_ids': plan['remaining_ids']}
    path = work / (stage + '.workflow.json')
    atomic(path, state)
    reserved = False
    try:
        if plan['remaining_ids']:
            ids = work / (stage + '.ids.txt')
            assert not ids.exists()
            ids.write_text(' '.join(map(str, plan['remaining_ids'])) + '\n')
            command = plan['original_command'].copy()
            command[command.index('--dagger-motion-ids') + 1] = str(ids)
            chain.training_cwd = HERE.parent / 'Training'
            chain.env = environment(0)
            chain.env.update(PYTHONPATH=str(chain.sim), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                             OPENBLAS_NUM_THREADS='1', PXR_WORK_THREAD_LIMIT='4', TOKENIZERS_PARALLELISM='false')
            devices = subprocess.check_output([sys.executable, str(HERE / 'gpu_slots.py'),
                       'reserve', '--owner', str(os.getpid()), '--count', '1', '--mb', '24000',
                       '--gpu', str(gpu)], text=True).split()
            assert devices == [str(gpu)]
            reserved = True
            state.update(status='collecting'); atomic(path, state)
            chain.execute(command, stage, gpu, chain.training_cwd)
            text = (work / (stage + '.log')).read_text()
            verify_collection_logs([text], plan['remaining_ids'])
            recovered = progress([text], plan['remaining_ids'])
            assert not recovered['remaining_ids']
        else:
            recovered = dict(written_ids=[], valid_skip_ids=[], completed_ids=[])
        written = sorted(plan['written_ids'] + recovered['written_ids'])
        assert sorted(plan['completed_ids'] + recovered['completed_ids']) == plan['expected_ids']
        inspect_recordings(work / 'raw' / name, written)
        episodes = audit_batch(work / 'raw' / name, plan['expected_ids'], 6.5)
        assert sorted(row['motion_id'] for row in episodes) == written
        report = {'student_attempts': 10, 'motion_ids': plan['expected_ids'], 'errors': 0,
                  'batch': name, 'episodes': episodes,
                  'valid_skip_ids': sorted(plan['valid_skip_ids'] + recovered['valid_skip_ids']),
                  'raw_directory': str(work / 'raw' / name), 'completed_at': time.time(),
                  'reboot_recovery': {'plan': name + '.' + SUFFIX + '.plan.json',
                     'reused_finalized_ids': plan['completed_ids'],
                     'reissued_infrastructure_interrupted_ids': plan['infrastructure_interrupted_ids'],
                     'new_log': stage + '.log', 'rng_note': plan['rng_note']}}
        atomic(work / (name + '.complete.json'), report)
        state.update(status='complete', completed_at=time.time(), report=report)
        atomic(path, state)
    except BaseException as error:
        state.update(status='failed', error=repr(error)); atomic(path, state)
        raise
    finally:
        if reserved:
            chain.release(gpu)
    if name == 'batch_002':
        os.execv(sys.executable, [sys.executable, str(HERE / 'walk_late_takeover_collect.py'),
                 '--tag', TAG, '--window-s', '6.5', '--batch-size', '10', '--attempts', '200', '--gpu', str(gpu)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'run'])
    parser.add_argument('--batch', type=int, choices=[2, 19], required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'plan':
        print(json.dumps(build_plan(args.batch)))
    else:
        assert args.gpu is not None
        plan = read(ROOT / 'dline' / TAG / (f'batch_{args.batch:03d}.{SUFFIX}.plan.json'))
        run(plan, args.gpu)


if __name__ == '__main__':
    main()
