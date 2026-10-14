"""Enforce one CPU mask for all Kevin workloads, regardless of agent/worktree.

The shared policy is ~/kevin/dline/resource_policy.json. CPU utilization is
measured as process CPU seconds / wall seconds / host logical CPU count.
Only this account's processes attributable to ~/kevin are changed.
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


def processes(root):
    rows = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            text = (directory/'stat').read_text()
            fields = text[text.rfind(')')+2:].split()
            args = (directory/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not args[0] or Path(args[0]).name == 'tmux':
                continue
            try:
                owned = (directory/'cwd').resolve().is_relative_to(root)
            except OSError:
                owned = False
            owned |= any(arg.startswith(str(root)+'/') for arg in args)
            rows[int(directory.name)] = {
                'ppid': int(fields[1]), 'ticks': int(fields[11])+int(fields[12]),
                'start': int(fields[19]), 'owned': owned,
            }
        except (OSError, ValueError, IndexError):
            continue
    # Include children which subsequently change their working directory.
    changed = True
    while changed:
        changed = False
        for row in rows.values():
            if not row['owned'] and rows.get(row['ppid'], {}).get('owned'):
                row['owned'] = changed = True
    return {pid: row for pid, row in rows.items() if row['owned']}


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
        rows = processes(root)
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
        cores = ticks/hz/elapsed if elapsed else None
        atomic(work/'cpu_share.json', {
            'time': time.time(), 'guard_pid': os.getpid(), 'available_cpus': cpus,
            'cpu_fraction_limit': fraction, 'shared_mask': sorted(mask), 'pids': sorted(rows),
            'changed_threads': changes, 'affinity_violations': violations, 'errors': errors,
            'sample_seconds': elapsed, 'measured_cpu_cores': cores,
            'measured_cpu_pct_host': 100*cores/cpus if cores is not None else None,
            'measurement_scope': 'live attributed processes; exited processes may be missed',
        })
        previous, previous_time = rows, now
        time.sleep(5)


if __name__ == '__main__':
    main()
