"""Read-only CPU accounting; never creates cgroups or changes any limits.

Use systemd's existing per-user kernel counter when every observed account
process belongs to it. Endpoint process deltas remain a diagnostic lower bound:
they omit children that start and exit entirely between the two snapshots.
"""
import os
import pathlib
import time


def account_snapshot(uid):
    rows, errors = {}, []
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        owned = False
        try:
            owned = path.stat().st_uid == uid
            if not owned:
                continue
            raw = (path / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            scopes = [line[3:] for line in (path / 'cgroup').read_text().splitlines()
                      if line.startswith('0::')]
            rows[int(path.name)] = {'ticks': int(fields[11]) + int(fields[12]),
                                    'born': fields[19], 'scope': scopes[0] if len(scopes) == 1 else None}
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, IndexError) as exc:
            if owned:
                errors.append(f'{path.name}: {exc!r}')
    return rows, errors


def kernel_counter(uid):
    path = pathlib.Path('/sys/fs/cgroup/user.slice') / f'user-{uid}.slice/cpu.stat'
    stat = path.stat()
    values = dict(line.split() for line in path.read_text().splitlines())
    return {'usage_usec': int(values['usage_usec']), 'identity': (stat.st_dev, stat.st_ino)}


def cpu_result(uid, before, after, first, last, elapsed, logical_cpus, hz, errors=()):
    scope = f'/user.slice/user-{uid}.slice'
    uncovered = sorted({pid for rows in (before, after) for pid, row in rows.items()
                        if row['scope'] != scope and not (row['scope'] or '').startswith(scope + '/')})
    ticks = sum(max(0, row['ticks'] - before[pid]['ticks'])
                for pid, row in after.items() if pid in before and row['born'] == before[pid]['born'])
    lower = ticks / hz / elapsed / logical_cpus * 100
    issues = list(errors)
    if uncovered:
        issues.append('Account processes outside the observed user slice: ' + str(uncovered))
    if not before or not after:
        issues.append('Account process coverage is empty')
    if first is None or last is None:
        issues.append('Per-user kernel CPU counter unavailable')
    elif first['identity'] != last['identity'] or last['usage_usec'] < first['usage_usec']:
        issues.append('Kernel counter was replaced or reset')
    result = {'sampled_at': time.time(), 'cpu_sample_seconds': elapsed,
              'logical_cpus': logical_cpus, 'process_snapshot_lower_bound_pct': lower,
              'observed_account_processes_before': len(before),
              'observed_account_processes_after': len(after),
              'cpu_measurement_complete': not issues,
              'cpu_accounting_scope': scope}
    if issues:
        result.update(account_cpu_pct=lower, cpu_measurement_source='process_snapshot_lower_bound_incomplete',
                      cpu_snapshot_error='; '.join(issues))
    else:
        kernel = (last['usage_usec'] - first['usage_usec']) / 1e6 / elapsed / logical_cpus * 100
        result.update(account_cpu_pct=max(kernel, lower), kernel_user_slice_cpu_pct=kernel,
                      cpu_measurement_source='kernel_user_slice_with_verified_process_coverage')
    return result


def sample_account_cpu(seconds=1):
    uid = os.getuid()
    before, errors = account_snapshot(uid)
    try:
        first = kernel_counter(uid)
    except (OSError, ValueError, KeyError) as exc:
        first = None
        errors.append(repr(exc))
    start = time.monotonic()
    time.sleep(seconds)
    try:
        last = kernel_counter(uid)
    except (OSError, ValueError, KeyError) as exc:
        last = None
        errors.append(repr(exc))
    elapsed = time.monotonic() - start
    after, final_errors = account_snapshot(uid)
    return cpu_result(uid, before, after, first, last, elapsed, os.cpu_count(),
                      os.sysconf('SC_CLK_TCK'), [*errors, *final_errors])
