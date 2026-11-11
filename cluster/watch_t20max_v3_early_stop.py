"""Authorized three-checkpoint patience with additional-seed confirmation.

Attaches to the existing v3 controller without restarting training. Only the
recorded, identity-checked controller and training PIDs can be stopped. Existing
evaluations finish first; all checkpoints remain on disk.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import watch_t20max_v3 as run

OUT = run.OUT / "early_stop"
POOL = "StitchMAX100v3_eval"
PATIENCE = 3
SEEDS = (1, 2)


def decision(rows, after=-1):
    rows = sorted((r for r in rows if r['model'] == run.RUN and r['pool'] == POOL),
                  key=lambda r: r['checkpoint_iteration'])
    assert len({r['checkpoint_iteration'] for r in rows}) == len(rows)
    best, count, previous = None, 0, None
    for row in rows:
        assert row['episodes'] >= 2000
        it = row['checkpoint_iteration']
        if best is None or row['held5_pct'] > best['held5_pct']:
            best, count = row, 0
        else:
            # Missing/failed evaluations cannot count as consecutive check-ins.
            count = count + 1 if previous is not None and it-previous == 500 else 1
        if it <= after:
            count = 0
        previous = it
    latest = rows[-1] if rows else None
    return {'best': best, 'latest': latest, 'non_improving_evals': count,
            'eligible': bool(latest and latest['checkpoint_iteration'] >= 5000 and count >= PATIENCE)}


def identity(pid, expected):
    p = Path('/proc', str(pid))
    assert p.stat().st_uid == os.getuid(), f'PID {pid} is not owned by this login'
    args = (p/'cmdline').read_bytes().decode().strip('\0').split('\0')
    assert args == expected, (pid, args, expected)
    # Also pin start time, preventing accidental signals to a reused PID.
    return (p/'stat').read_text().rsplit(')', 1)[1].split()[19]


def send(pid, expected, start, sig):
    assert identity(pid, expected) == start
    os.kill(pid, sig)


def checkpoint_receipt(path):
    import torch
    path = Path(path)
    assert path.is_relative_to(run.ROOT) and path.is_file()
    assert time.time()-path.stat().st_mtime > 10, 'Checkpoint may still be writing'
    data = torch.load(path, map_location='cpu', weights_only=False)
    assert data['model_state_dict'] and data['optimizer_state_dict']['state']
    assert 'iter' in data
    return {'path': str(path), 'iteration': int(data['iter']),
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'has_model_and_optimizer': True}


def stop_owned_training(state, check, confirmation):
    """Freeze scheduling, check for unreported eval launches, then stop our run."""
    train_pid, controller_pid = state['train_pid'], state['controller_pid']
    launch = json.loads((run.OUT/'train.launch.json').read_text())
    assert train_pid == launch['pid']
    train_args = launch['command']
    assert '--run_name' in train_args and train_args[train_args.index('--run_name')+1] == run.RUN
    controller_args = [str(run.PYTHON), '-u', 'cluster/watch_t20max_v3.py']
    cstart = identity(controller_pid, controller_args)
    tstart = identity(train_pid, train_args)
    send(controller_pid, controller_args, cstart, signal.SIGSTOP)
    controller_stopped = False
    try:
        children = Path('/proc', str(controller_pid), 'task', str(controller_pid), 'children').read_text().split()
        # An eval could have launched just before SIGSTOP but not reached workflow.json.
        for child in children:
            if int(child) == train_pid:
                continue
            p = Path('/proc', child, 'cmdline')
            if p.exists() and p.read_bytes():
                return False
        current = json.loads((run.OUT/'workflow.json').read_text())
        if current.get('active_eval') or current.get('pending_evals'):
            return False
        fresh = decision(json.loads((run.OUT/'results.json').read_text()))
        if not fresh['eligible'] or fresh['latest']['checkpoint_iteration'] != check['latest']['checkpoint_iteration']:
            return False
        directory = Path(current['checkpoint_dir'])
        assert directory.name.endswith('_'+run.RUN) and directory.is_relative_to(run.ROOT)
        candidates = sorted(directory.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
        stable = next(p for p in candidates if time.time()-p.stat().st_mtime > 10)
        retained = {'best_evaluated': checkpoint_receipt(check['best']['checkpoint']),
                    'latest_resumable': checkpoint_receipt(stable)}
        receipt = {'status': 'stop_requested', 'requested_at': time.time(),
                   'rule': 'Three consecutive held-out checkpoints without a new best; two extra seeds confirm no improvement',
                   'decision': check, 'confirmation': confirmation, 'preserved': retained,
                   'controller_pid': controller_pid, 'train_pid': train_pid,
                   'last_logged_iteration': current.get('latest_iteration'),
                   'note': 'All checkpoints retained. Resume from the latest verified optimizer checkpoint; unsaved updates may be lost.'}
        for label, item in retained.items():
            link = OUT/(label+'.pt')
            assert not link.exists() and not link.is_symlink()
            link.symlink_to(os.path.relpath(item['path'], OUT))
        run.write(OUT/'stop_receipt.json', receipt)
        send(controller_pid, controller_args, cstart, signal.SIGTERM)
        send(controller_pid, controller_args, cstart, signal.SIGCONT)
        controller_stopped = True
        send(train_pid, train_args, tstart, signal.SIGTERM)
        # No force kill: if shutdown hangs, expose an alert for review.
        deadline = time.time()+60
        while time.time() < deadline:
            alive = []
            for pid in [controller_pid, train_pid]:
                p = Path('/proc', str(pid), 'stat')
                if p.exists() and p.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                    alive.append(pid)
            if not alive:
                break
            time.sleep(2)
        assert not alive, f'Shutdown pending for owned PIDs {alive}; no force kill issued'
        receipt.update(status='early_stopped', completed_at=time.time())
        run.write(OUT/'stop_receipt.json', receipt)
        current.update(status='early_stopped', early_stop_receipt=str(OUT/'stop_receipt.json'),
                       heartbeat=time.time(), active_eval=None, pending_evals=0)
        run.write(run.OUT/'workflow.json', current)
        return True
    finally:
        if not controller_stopped:
            send(controller_pid, controller_args, cstart, signal.SIGCONT)


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    state = {'status': 'monitoring', 'pid': os.getpid(), 'patience': PATIENCE,
             'metric': 'held5_pct', 'pool': POOL, 'confirmation_seeds': list(SEEDS),
             'minimum_checkpoint': 5000, 'alerts': [], 'started': time.time()}
    results, active, defer_after = {}, None, -1
    try:
        while True:
            state['heartbeat'] = time.time()
            workflow = json.loads((run.OUT/'workflow.json').read_text())
            if workflow['status'] in ['complete', 'early_stopped', 'failed']:
                state['status'] = 'run_'+workflow['status']
                run.write(OUT/'status.json', state)
                return
            if time.time()-workflow['heartbeat'] > 120:
                raise RuntimeError('Main controller heartbeat is stale; no automatic stop')
            check = decision(json.loads((run.OUT/'results.json').read_text()), defer_after)
            state.update(decision=check, status='monitoring', alerts=[])
            if active:
                key, proc, log, started = active
                if proc.poll() is not None:
                    assert proc.returncode == 0, f'Confirmation eval failed: {log}'
                    results[key] = run.metrics(run.tail(log, 2000000))
                    run.write(OUT/'confirmation_results.json', [{'checkpoint_iteration':k[0], 'seed':k[1], **v} for k,v in results.items()])
                    active = None
                elif time.time()-started > 1800 or 'Traceback (most recent call last)' in run.tail(log):
                    raise RuntimeError(f'Confirmation evaluation stalled or failed: {log}')
            if check['eligible']:
                state['status'] = 'confirming_plateau'
                pair = [check['best'], check['latest']]
                missing = [(row, seed) for row in pair for seed in SEEDS
                           if (row['checkpoint_iteration'], seed) not in results]
                if not missing and active is None:
                    rates = []
                    for row in pair:
                        group = [results[(row['checkpoint_iteration'], s)] for s in SEEDS]
                        rates.append(100*sum(x['held5_successes'] for x in group)/sum(x['episodes'] for x in group))
                    confirmation = {'best_extra_seed_success_pct':rates[0], 'latest_extra_seed_success_pct':rates[1],
                                    'seeds':list(SEEDS), 'supports_stopping':rates[1] <= rates[0]}
                    state['confirmation'] = confirmation
                    if not confirmation['supports_stopping']:
                        defer_after = check['latest']['checkpoint_iteration']
                        state.update(status='continued_after_confirmation_improvement', defer_after=defer_after)
                    elif not workflow.get('active_eval') and not workflow.get('pending_evals'):
                        if stop_owned_training(workflow, check, confirmation):
                            state.update(status='early_stopped', completed=time.time())
                            run.write(OUT/'status.json', state)
                            return
                    else:
                        state['status'] = 'waiting_for_scheduled_evals_before_stop'
                elif active is None and run.gpu_free(8):
                    row, seed = missing[0]
                    key = (row['checkpoint_iteration'], seed)
                    log = OUT/f'confirm_{key[0]}_seed{seed}.log'
                    assert not log.exists(), f'Refusing to overwrite {log}'
                    cmd = run.command('eval_sonic_adapter.py', POOL) + ['--num_envs','1000',
                        '--num_episodes','2000','--seed',str(seed),'--path',row['checkpoint']]
                    with log.open('w') as f:
                        proc = subprocess.Popen(cmd, cwd=run.REPO/'Training', env=run.environment(8), stdout=f, stderr=subprocess.STDOUT)
                    active = (key, proc, log, time.time())
                    run.write(log.with_suffix('.launch.json'), {'pid':proc.pid,'command':cmd,'gpu':8,'started':time.time()})
            state['active_confirmation'] = {'checkpoint':active[0][0], 'seed':active[0][1], 'pid':active[1].pid} if active else None
            run.write(OUT/'status.json', state)
            time.sleep(15)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), heartbeat=time.time(),
                     alerts=['Early-stop watcher failed; training is not automatically stopped.'])
        run.write(OUT/'status.json', state)
        raise


if __name__ == '__main__':
    main()
