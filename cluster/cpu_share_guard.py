"""Enforce one CPU mask for all jobs under the shared cluster login.

The shared policy is ~/kevin/dline/resource_policy.json. CPU utilization is
measured as process CPU seconds / wall seconds / host logical CPU count.
The account-wide scope includes jobs outside ~/kevin. GPU blocking reservations
also make older launchers using the shared reservation file obey the device cap.
"""
import fcntl
import json
import os
from pathlib import Path
import time


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def processes():
    rows = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            text = (directory/'stat').read_text()
            fields = text[text.rfind(')')+2:].split()
            args = (directory/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not args[0]:
                continue
            rows[int(directory.name)] = {
                'ppid': int(fields[1]), 'ticks': int(fields[11])+int(fields[12]),
                'start': int(fields[19]),
            }
        except (OSError, ValueError, IndexError):
            continue
    return rows


def sync_gpu_policy(work, policy):
    """Block disallowed devices for every launcher using gpu_slots.json."""
    allowed = set(policy['allowed_gpus'])
    devices = set(policy['all_gpu_indices'])
    assert allowed <= devices and len(allowed) <= policy['max_gpu_count']
    with (work/'gpu_slots.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = work/'gpu_slots.json'
        entries = json.loads(path.read_text()) if path.exists() else []
        entries = [row for row in entries if not row.get('resource_policy_block')
                   and (Path('/proc')/str(row['owner'])).exists()]
        # Legacy launchers discard reservations whose owner PID disappears.
        # PID 1 is a persistent bookkeeping sentinel, not a job we manage. This
        # keeps the policy blocks in place even if this guard exits unexpectedly.
        entries.extend({'owner': 1, 'gpu': gpu, 'mb': 10**9,
                        'resource_policy_block': True, 'policy_guard_pid': os.getpid()}
                       for gpu in sorted(devices-allowed))
        atomic(path, entries)


def main():
    root = (Path.home()/'kevin').resolve()
    work = root/'dline'
    work.mkdir(exist_ok=True)
    lock = (work/'cpu_share_guard.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = os.cpu_count()
    hz = os.sysconf('SC_CLK_TCK')
    previous, previous_time = {}, None
    while True:
        policy = json.loads((work/'resource_policy.json').read_text())
        fraction = policy['cpu_fraction']
        mask = set(policy['shared_mask'])
        assert 0 < fraction <= 0.20 and 0 < len(mask) <= int(cpus*fraction)
        assert all(isinstance(cpu, int) and 0 <= cpu < cpus for cpu in mask)
        now = time.monotonic()
        rows = processes()
        elapsed = now-previous_time if previous_time is not None else None
        ticks = sum(max(0, row['ticks']-previous[pid]['ticks']) for pid, row in rows.items()
                    if pid in previous and row['start'] == previous[pid]['start'])
        violations, changes, errors = [], [], []
        for pid in rows:
            try:
                for thread in (Path('/proc')/str(pid)/'task').iterdir():
                    tid = int(thread.name)
                    before = os.sched_getaffinity(tid)
                    if before != mask:
                        os.sched_setaffinity(tid, mask)
                        changes.append(tid)
                    if os.sched_getaffinity(tid)-mask:
                        violations.append(tid)
            except ProcessLookupError:
                continue
            except OSError as exc:
                errors.append({'pid': pid, 'error': str(exc)})
        sync_gpu_policy(work, policy)
        cores = ticks/hz/elapsed if elapsed else None
        atomic(work/'cpu_share.json', {
            'time': time.time(), 'guard_pid': os.getpid(), 'available_cpus': cpus,
            'cpu_fraction_limit': fraction, 'shared_mask': sorted(mask), 'pids': sorted(rows),
            'changed_threads': changes, 'affinity_violations': violations, 'errors': errors,
            'sample_seconds': elapsed, 'measured_cpu_cores': cores,
            'measured_cpu_pct_host': 100*cores/cpus if cores is not None else None,
            'account_uid': os.getuid(), 'accounting_scope': 'all jobs under this login, regardless of directory',
            'allowed_gpus': policy['allowed_gpus'], 'max_gpu_count': policy['max_gpu_count'],
            'measurement_scope': 'live account processes; exited processes may be missed',
        })
        previous, previous_time = rows, now
        time.sleep(5)


if __name__ == '__main__':
    main()
