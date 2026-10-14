"""Run four unchanged GR00T evaluation shards on Biped GPU 0 in sequence."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import watch_t20max_v3 as common


def main(work):
    # Apply the operator's per-job allocation before spawning any eval workers.
    # This affects this controller and its children, never other account jobs.
    allocation = common.ROOT/'dline/cpu_share.json'
    if allocation.exists():
        os.sched_setaffinity(0, json.loads(allocation.read_text())['shared_mask'])
    spec = json.loads((work/'request.json').read_text())
    run = spec['run']
    assert re.fullmatch(r'walk(10|50|200)_v3_dagger57_20261013_r[0-3]', run)
    assert spec['pool'] == 'StitchMAX100v3_eval' and spec['episodes'] == 100
    out = common.ROOT/'eval_videos'/run
    assert not out.exists(), 'Reconcile prior evaluation before restarting'
    out.mkdir(parents=True)
    lock = (work/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mapping = spec['reference_mapping']
    common.write(out/'reference_mapping.json', mapping)
    common.write(out/'box.json', spec['box'])
    native = {r['canonical_id']: r['source_id'] for r in mapping['references']}
    assert sorted(native) == sorted(native.values()) == list(range(100))
    checkpoint = common.ROOT/spec['checkpoint_relative']
    assert checkpoint.is_dir() and checkpoint.resolve().is_relative_to(common.ROOT.resolve())
    state = {'status': 'running', 'controller_pid': os.getpid(), 'run': run,
             'started': time.time(), 'completed_shards': 0, 'gpu': 0, 'errors': [], 'active_job': None,
             'allocated_cpus': sorted(os.sched_getaffinity(0))}
    try:
        for shard in range(4):
            slots = []
            while not slots:
                slots = subprocess.check_output([str(common.PYTHON), str(common.REPO/'cluster/gpu_slots.py'),
                    'reserve', '--owner', str(os.getpid()), '--count', '1', '--mb', '17000', '--gpu', '0'], text=True).split()
                assert not slots or slots == ['0']
                state.update(status='waiting_for_gpu' if not slots else 'running', heartbeat=time.time())
                common.write(work/'workflow.json', state)
                if not slots:
                    time.sleep(20)
            try:
                ids = work/f'ids{shard}.txt'
                ids.write_text(' '.join(str(native[i]) for i in range(shard, 100, 4))+'\n')
                cmd = [str(common.PYTHON), str(common.REPO/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'),
                    '--headless', '--sonic-pt', str(common.ROOT/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'),
                    '--ref-motions-path', spec['pool'], '--vla-checkpoint', str(checkpoint), '--waist-dof', '29',
                    '--motions-from', str(ids), '--num-episodes', '25', '--seed', str(100+shard),
                    '--traj-dump', str(out/f'b{shard}'), '--max-steps-per-episode', '1200']
                logfile = out/f'eval_{run}_shard{shard}.log'
                assert not logfile.exists()
                env = common.environment(0)
                env.update(HS_EVAL_NO_EE_TERM='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                with logfile.open('w') as stream:
                    proc = subprocess.Popen(cmd, cwd=common.REPO/'Training', env=env,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    state['active_job'] = {'pid': proc.pid, 'shard': shard, 'log': str(logfile), 'command': cmd}
                    common.write(work/f'shard{shard}.launch.json', state['active_job'])
                    while proc.poll() is None:
                        state.update(heartbeat=time.time(), log_age=time.time()-logfile.stat().st_mtime)
                        common.write(work/'workflow.json', state)
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            pass
                assert proc.returncode == 0, f'Evaluation exited {proc.returncode}: {logfile}'
                assert 'VLA + SONIC EVAL SUMMARY' in common.tail(logfile), str(logfile)
                assert '[camera-check]' in logfile.read_text(errors='replace'), 'Camera check did not run'
                state.update(completed_shards=shard+1, active_job=None, heartbeat=time.time())
                common.write(work/'workflow.json', state)
            finally:
                subprocess.run([str(common.PYTHON), str(common.REPO/'cluster/gpu_slots.py'),
                    'release', '--owner', str(os.getpid()), '--gpu', '0'], check=True)
        with (work/'score.log').open('w') as stream:
            subprocess.run([str(common.PYTHON), str(common.REPO/'cluster/walk_gr00t_score.py'), str(out), str(out/'box.json')],
                cwd=common.REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((out/'results.json').read_text())
        assert result['episodes'] == 100 and sorted(r['motion_id'] for r in result['per_episode']) == list(range(100))
        state.update(status='complete', completed_at=time.time(), heartbeat=time.time(), output=str(out))
        common.write(work/'workflow.json', state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), heartbeat=time.time())
        common.write(work/'workflow.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    main(parser.parse_args().work)
