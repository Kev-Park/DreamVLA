"""Evaluate the selected direct-token checkpoint on the residual comparison seeds."""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import continue_walk_direct_biped as direct
import watch_t20max_v3 as common

OUT = common.ROOT/'eval_videos/walk_v3_study/direct_matched4500_seeds'
CHECKPOINTS, SEEDS = [4500], [1, 2]


def coverage(text, episodes):
    ranking = text.split('PER-CLIP OUTCOME RANKING (worst topple first)')[-1]
    counts = [(int(i), int(n)) for i, n in re.findall(
        r'^\s+(\d+)\s+(\d+)\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+', ranking, re.M)]
    return sorted(i for i, n in counts) == list(range(100)) and sum(n for i, n in counts) == episodes and episodes >= 2000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'workflow.json').exists(), 'Inspect previous comparison before restarting'
    primary = json.loads((direct.OUT/'workflow.json').read_text())
    assert primary['status'] == 'paused_by_user' and primary['pause']['selected_iteration'] == 4500
    directory = Path(primary['checkpoint_dir'])
    assert directory.is_relative_to(common.REPO/'Training/logs/rsl_rl/g1_sonic_adapter')
    for iteration in CHECKPOINTS:
        assert (directory/f'model_{iteration}.pt').is_file()
    pending = [(iteration, seed) for seed in SEEDS for iteration in CHECKPOINTS]
    active, rows, errors = [], [], []
    state = {'status': 'validating', 'controller_pid': os.getpid(), 'started': time.time(),
             'checkpoint_dir': str(directory), 'checkpoints': CHECKPOINTS, 'seeds': SEEDS,
             'target_episodes': 2000, 'pool': 'StitchMAX100v3_eval',
             'training_unchanged': True, 'errors': errors}
    try:
        while pending or active:
            while pending and len(active) < 2:
                # Restrict these checks to our existing training GPU. Startup
                # reservations also protect other Kevin workflows on this host.
                slots = subprocess.check_output([str(common.PYTHON), str(common.REPO/'cluster/gpu_slots.py'),
                    'reserve', '--owner', str(os.getpid()), '--count', '1', '--mb', '6000', '--gpu', '0'], text=True).split()
                if not slots:
                    break
                assert slots == ['0']
                iteration, seed = pending.pop(0)
                name = f'eval_{iteration}_seed{seed}'
                log = OUT/(name+'.log')
                assert not log.exists()
                cmd = direct.command('eval_sonic_adapter.py', 'StitchMAX100v3_eval') + [
                    '--num_envs', '1000', '--num_episodes', '2000', '--seed', str(seed),
                    '--path', str(directory/f'model_{iteration}.pt')]
                with log.open('w') as stream:
                    proc = subprocess.Popen(cmd, cwd=common.REPO/'Training', env=common.environment(0),
                                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                job = {'iteration': iteration, 'seed': seed, 'pid': proc.pid, 'started': time.time(),
                       'log': str(log), 'gpu': 0, 'command': cmd}
                common.write(OUT/(name+'.launch.json'), job)
                active.append((job, proc, log))
            for entry in active[:]:
                job, proc, log = entry
                if proc.poll() is None:
                    continue
                subprocess.run([str(common.PYTHON), str(common.REPO/'cluster/gpu_slots.py'),
                    'release', '--owner', str(os.getpid()), '--gpu', '0'], check=True)
                try:
                    assert proc.returncode == 0, f'Exit {proc.returncode}: {log}'
                    text = common.tail(log, 2000000)
                    row = dict(iteration=job['iteration'], seed=job['seed'], **common.metrics(text))
                    assert coverage(text, row['episodes']), f'Incomplete reference coverage: {log}'
                    assert all(math.isfinite(row[k]) for k in ['held5_pct', 'mpjpe_world_cm', 'mpjpe_root_relative_cm'])
                    row['all_100_references_covered'] = True
                    rows.append(row)
                    common.write(OUT/'results.json', rows)
                except Exception as exc:
                    errors.append({'job': job, 'error': repr(exc)})
                active.remove(entry)
            state.update(heartbeat=time.time(), completed=len(rows), errors=errors,
                         waiting=len(pending), active_jobs=[dict(j, log_age=time.time()-p.stat().st_mtime) for j, proc, p in active])
            common.write(OUT/'workflow.json', state)
            if pending or active:
                time.sleep(20)
        assert len(rows) == 2 and not errors, errors
        original = json.loads((direct.OUT/'results.json').read_text())
        summary = {}
        for iteration in CHECKPOINTS:
            selected = [dict(r, seed=0) for r in original if r['iteration'] == iteration]
            selected += [r for r in rows if r['iteration'] == iteration]
            assert sorted(r['seed'] for r in selected) == [0, 1, 2]
            summary[str(iteration)] = {'episodes': sum(r['episodes'] for r in selected),
                'successes': sum(r['held5_successes'] for r in selected),
                'seed_mean_success_pct': sum(r['held5_pct'] for r in selected)/3,
                'per_seed': {str(r['seed']): r['held5_pct'] for r in selected},
                'pooled_frame_mpjpe_world_cm': sum(r['mpjpe_world_cm']*r['mpjpe_frames'] for r in selected)/sum(r['mpjpe_frames'] for r in selected),
                'pooled_frame_mpjpe_root_relative_cm': sum(r['mpjpe_root_relative_cm']*r['mpjpe_frames'] for r in selected)/sum(r['mpjpe_frames'] for r in selected)}
        common.write(OUT/'comparison.json', {'summary': summary, 'seeds': [0, 1, 2],
                                           'training_unchanged': True})
        state.update(status='complete', completed_at=time.time(), summary=summary)
        common.write(OUT/'workflow.json', state)
    except BaseException as exc:
        state.update(status='failed', heartbeat=time.time(), error=repr(exc))
        common.write(OUT/'workflow.json', state)
        raise


if __name__ == '__main__':
    main()
