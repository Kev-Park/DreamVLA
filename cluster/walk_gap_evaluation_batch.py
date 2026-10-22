"""Run the four reviewed walking-gap evals in two fixed GPU allocations."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from gr00t_budget_chain import atomic, checkpoint_ok, read
from walk_gr00t_chain import WalkChain
from watch_walk_gr00t import output_activity, process_argv, stage_command_matches


def main():
    assert socket.gethostname().startswith('bluesclues')
    root = Path.home() / 'kevin'
    repo = Path(__file__).resolve().parent.parent
    work = root / 'dline/walk_gap_ablation20261022'
    review = read(work / 'operator_review.json')
    assert review['user_approval_received'] and review['user_approval_reference']
    assert review['approved_conditions'] == ['start_at_reference_frame10', 'reference_scheduled_close']
    assert review['decoder_contract'] == 'previous_full_fsq'
    assert review['account_resources_manually_reviewed'] and review['other_agents_pending_allocations_manually_reviewed']
    assert 0 <= time.time() - review['at'] < 120
    assert set(os.sched_getaffinity(0)) == set(review['cpus']) and len(review['cpus']) <= 18
    assert review['gpus'] == [0, 1] and review['eval_workers_per_gpu'] == 2
    assert review['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert review['free_disk_bytes'] > 5 * 2**30
    for relative, expected in review['source_sha256'].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected, relative
    assert review['source_sha256']['Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'] == '30a4cc7a7e15bb6df724f4a55cb1d14134f03715f64758c64e6a57f40b59f63c'
    assert review['source_sha256']['Training/scripts/reinforcement_learning/rsl_rl/vla_sonic/token_action_wrapper.py'] == '19a4288b0a61094b4711163ce6ee648cb060f2745445e6a6a182cc53279b5dbe'
    jobs = review['jobs']
    assert len(jobs) == 4 and len({j['tag'] for j in jobs}) == 4
    assert {(j['model'], j['condition']) for j in jobs} == {
        (m, c) for m in ['historical_r2', 'current_r2'] for c in review['approved_conditions']}
    for job in jobs:
        assert job['condition'] in review['approved_conditions'] and job['gpu'] in review['gpus']
        assert job['tag'].startswith('walk_gap_') and job['tag'].replace('_', '').isalnum()
        checkpoint = root / job['checkpoint_relative_to_kevin']
        assert checkpoint.resolve().is_relative_to(root.resolve()) and checkpoint_ok(checkpoint)
        assert not (root / 'eval_videos' / job['tag']).exists()
        assert not (root / 'dline' / job['tag']).exists()
    lock = (work / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (work / 'launch.json').exists()
    atomic(work / 'launch.json', {'at': time.time(), 'pid': os.getpid(), 'operator_review': review})
    slots = []
    stop = threading.Event()
    jobs_lock = threading.Lock()
    stage_states = {}

    class Diagnostic(WalkChain):
        def __init__(self, job):
            super().__init__(200)
            self.job = job
            self.tag = job['tag']
            self.work = root / 'dline' / self.tag
            self.work.mkdir()
            self.eval_workers = 2
            self.sim_task_threads = 4
            self.env['PXR_WORK_THREAD_LIMIT'] = '4'
            (self.work / 'box.json').write_bytes((root / 'dline/walk200_v3_dagger57_20261013/box.json').read_bytes())

        def external(self, phase, run, payload):
            assert phase == 'evaluate'
            return False

        def slots(self, count, mb):
            assert count == 2 and mb == 17000
            entries = read(root / 'dline/gpu_slots.json')
            owned = [e for e in entries if e['owner'] == os.getpid() and e['gpu'] == self.job['gpu']]
            assert len(owned) == 2 and all(e['mb'] == 17000 for e in owned)
            return [self.job['gpu']] * 2

        def release(self, gpu):
            # The batch owns these fixed allocations through both sequential conditions.
            assert gpu == self.job['gpu']

        def execute(self, args, label, gpu=None, cwd=None):
            if any(Path(str(a)).name == 'eval_vla_sonic.py' for a in args):
                assert '--skip-start-frames' not in args and '--force-close-at-grab' not in args
                args = [*args, '--skip-start-frames', '10'] if self.job['condition'] == 'start_at_reference_frame10' else [*args, '--force-close-at-grab']
            return super().execute(args, label, gpu, cwd)

    def health(status):
        workers, alerts = [], []
        for job in jobs:
            directory = root / 'dline' / job['tag']
            for path in directory.glob('*.process.json'):
                handle = read(path)
                live = stage_command_matches(handle['command'], process_argv(handle['pid']))
                row = {'tag': job['tag'], 'pid': handle['pid'], 'gpu': handle.get('gpu'), 'live': live}
                if live:
                    log = path.with_name(path.name.replace('.process.json', '.log'))
                    activity, stalled = output_activity(handle['pid'], log, time.time())
                    row.update(activity)
                    alerts.extend('Stalled: ' + x for x in stalled)
                workers.append(row)
        with jobs_lock:
            states = dict(stage_states)
        alerts.extend(tag + ': ' + state['error'] for tag, state in states.items() if 'error' in state)
        atomic(work / 'workflow.json', {'heartbeat': time.time(), 'pid': os.getpid(), 'status': status,
            'workers': workers, 'stages': states, 'alerts': alerts, 'gpus': review['gpus'],
            'decoder_contract': 'previous_full_fsq', 'fixed_allocations_retained_until_batch_exit': True})

    def run_model(gpu):
        for job in [j for j in jobs if j['gpu'] == gpu]:
            if stop.is_set():
                with jobs_lock:
                    stage_states[job['tag']] = {'status': 'held_after_batch_failure'}
                return
            try:
                chain = Diagnostic(job)
                atomic(chain.work / 'diagnostic_recipe.json', {'at': time.time(), 'batch_pid': os.getpid(),
                    'job': job, 'operator_review': review, 'decoder_contract': 'previous_full_fsq'})
                with jobs_lock:
                    stage_states[job['tag']] = {'status': 'evaluating', 'started': time.time()}
                chain.evaluate(job['tag'], root / job['checkpoint_relative_to_kevin'])
                report_path = root / 'eval_videos' / job['tag'] / 'results.json'
                report = read(report_path)
                assert report['episodes'] == 100 and report['pool'] == 'StitchMAX100v3_eval'
                assert sorted(r['motion_id'] for r in report['per_episode']) == list(range(100))
                done = {'at': time.time(), 'status': 'complete', 'condition': job['condition'],
                    'results': str(report_path), 'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    'raw_trajectory_audit_pending': True}
                atomic(chain.work / 'done.json', done)
                with jobs_lock:
                    stage_states[job['tag']] = done
            except BaseException as exc:
                stop.set()
                with jobs_lock:
                    stage_states[job['tag']] = {'status': 'failed_for_review', 'error': repr(exc)}
                atomic(root / 'dline' / job['tag'] / 'abort.json', {'at': time.time(), 'error': repr(exc)})
                raise

    try:
        for gpu in review['gpus']:
            selected = list(map(int, subprocess.check_output([sys.executable, str(repo / 'cluster/gpu_slots.py'),
                'reserve', '--owner', str(os.getpid()), '--count', '2', '--mb', '17000', '--gpu', str(gpu)], text=True).split()))
            slots.extend(selected)
            assert selected == [gpu, gpu], 'Reviewed capacity changed before allocation'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_model, gpu) for gpu in review['gpus']]
            while not all(f.done() for f in futures):
                health('needs_attention' if stop.is_set() else 'evaluating')
                concurrent.futures.wait(futures, timeout=30)
            for future in futures:
                future.result()
        assert all(stage_states[j['tag']]['status'] == 'complete' for j in jobs)
        atomic(work / 'done.json', {'at': time.time(), 'jobs': stage_states, 'count': 4})
        health('complete')
    except BaseException as exc:
        atomic(work / 'abort.json', {'at': time.time(), 'pid': os.getpid(), 'error': repr(exc)})
        health('failed_for_review')
        raise
    finally:
        for gpu in slots:
            subprocess.run([sys.executable, str(repo / 'cluster/gpu_slots.py'), 'release',
                '--owner', str(os.getpid()), '--gpu', str(gpu)], check=True)


if __name__ == '__main__':
    main()
