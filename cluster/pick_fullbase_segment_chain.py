"""Read authoritative terminal states; submit each continuation once, then final eval."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from pick_fullbase_segment import TOTAL_STEPS, checkpoint_step, write_json


def scheduler_state(job):
    p = subprocess.run(['sacct', '-X', '-j', str(job), '-n', '-P',
                        '--format=JobID,State,ExitCode'], capture_output=True, text=True, timeout=15)
    if p.returncode:
        raise RuntimeError(p.stderr)
    rows = [line.split('|') for line in p.stdout.splitlines()]
    return next((row[1].split()[0] for row in rows if row[0] == str(job)), None)


def submit(state, path, kind, command):
    # Persist intent first. A lost response must be reconciled, never blindly resubmitted.
    state.update(phase='submitting_' + kind, submission_intent=command, updated_at=time.time())
    write_json(path, state)
    p = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError('Submission failed: ' + p.stderr)
    job = p.stdout.strip().split(';')[0]
    assert job.isdigit(), p.stdout
    state.pop('submission_intent')
    state[kind + '_job_id'] = job
    if kind == 'training':
        state['training_job_ids'].append(job)
        state['phase'] = 'training'
    else:
        state['phase'] = 'evaluation'
    state['updated_at'] = time.time()
    write_json(path, state)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    root = Path.home() / 'kevin'
    work = root / 'dline' / args.run
    path = work / 'segment_chain.json'
    with (work / 'segment_chain.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(path.read_text())
        assert state['run'] == args.run
        assert not state.get('submission_intent'), 'Unknown previous submission outcome needs reconciliation'
        assert state['phase'] in ['training', 'evaluation']
        try:
            while True:
                kind = 'evaluation' if state['phase'] == 'evaluation' else 'training'
                job = state[kind + '_job_id']
                status = scheduler_state(job)
                state.update(last_poll=time.time(), scheduler_state=status)
                write_json(path, state)
                if status == 'COMPLETED':
                    if kind == 'evaluation':
                        assert (work / 'done.json').is_file()
                        state.update(phase='done', updated_at=time.time())
                        write_json(path, state)
                        return
                    receipt = json.loads((work / f"segment-{state['segment']:03d}.json").read_text())
                    assert receipt['job_id'] == job
                    step = checkpoint_step(receipt['checkpoint'])
                    assert receipt['end_step'] == step > state['completed_steps']
                    state['completed_steps'] = step
                    if step == TOTAL_STEPS:
                        assert (work / 'trained.json').is_file()
                        command = ['sbatch', '--parsable', '--dependency=afterok:' + job,
                                   '--kill-on-invalid-dep=yes', '--chdir', str(work),
                                   '--output', str(work / 'eval.slurm-%j.out'),
                                   str(root / 'DreamVLA/cluster/adroit/pick_fullbase_batch16_eval.sbatch'),
                                   str(work / 'request.json')]
                        submit(state, path, 'evaluation', command)
                    else:
                        state['segment'] += 1
                        assert state['segment'] <= 32, 'Unexpectedly slow or short segments require review'
                        command = ['sbatch', '--parsable', '--chdir', str(work),
                                   '--output', str(work / 'train.segment-%j.out'),
                                   str(root / 'DreamVLA/cluster/adroit/pick_fullbase_segment.sbatch'),
                                   state['dataset'], args.run, str(state['segment'])]
                        submit(state, path, 'training', command)
                elif status in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                                'BOOT_FAIL', 'PREEMPTED', 'DEADLINE', 'REVOKED'}:
                    raise RuntimeError(f'{kind} {job} ended {status}; no automatic restart')
                time.sleep(45)
        except Exception as error:
            state.update(phase='needs_attention', error=repr(error), updated_at=time.time())
            write_json(path, state)
            raise


if __name__ == '__main__':
    main()
