"""Run the same experiment's unchanged R2 control on its RTX evaluation host."""
import argparse
import json
import os
from pathlib import Path
import time

import walk_late_takeover_bridge as bridge
from walk_adroit_bridge import save

RUN = 'walk200_r2_biped_control_20261026'
CHECKPOINT = 'checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--shared-state-dir', required=True)
    args = parser.parse_args()
    directory = Path(args.state_dir).resolve()
    directory.mkdir(exist_ok=True)
    bridge.configure(args.shared_state_dir)
    lock = (directory / 'control.lock').open('a+b')
    awake = None
    if os.name == 'nt':
        import ctypes, msvcrt
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        awake = ctypes.WinDLL('kernel32').SetThreadExecutionState
        awake(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    file = directory / 'workflow.json'
    state = {'started': time.time(), 'pid': os.getpid(), 'run': RUN,
             'experiment': bridge.TAG, 'checkpoint': CHECKPOINT,
             'host': 'biped', 'replaces_failed_adroit_job': '3431297'}

    def record(**values):
        state.update(heartbeat=time.time(), **values)
        save(file, state)

    try:
        while True:
            collector = bridge.collector_health()
            record(status='waiting_for_collection', collector=collector)
            if collector['status'] == 'collection_complete':
                break
            time.sleep(30)
        while True:
            launch = bridge.start_biped_evaluation(CHECKPOINT, run_name=RUN)
            record(status='waiting_for_resources', launch=launch)
            if not any(launch.get(k) for k in ('waiting_for_cpu_headroom', 'waiting_for_gpu_headroom')):
                break
            time.sleep(30)
        record(status='evaluating')
        started = time.time()
        while True:
            results = bridge.read_remote('biped', bridge.WORK + '/' + RUN + '.eval.json')
            workflow = bridge.read_remote('biped', bridge.WORK + '/evaluate_' + RUN + '.workflow.json')
            record(evaluation_state=workflow)
            assert workflow is not None or time.time() - started < 90
            assert not workflow or workflow['status'] != 'failed', workflow
            if results is not None:
                break
            if workflow and workflow['status'] == 'running':
                live = bridge.call('biped', "import json,pathlib\nprint(json.dumps(pathlib.Path('/proc',str(PID)).exists()))".replace('PID', repr(workflow['pid'])))
                assert live, ('Control evaluation controller disappeared', workflow)
            time.sleep(30)
        assert results['episodes'] == 100
        assert sorted(e['motion_id'] for e in results['per_episode']) == list(range(100))
        save(directory / 'results.json', results)
        record(status='complete', completed=time.time(), results=results)
    except BaseException as exc:
        record(status='stopped_for_review', error=repr(exc))
        raise
    finally:
        if awake:
            awake(0x80000000)


if __name__ == '__main__':
    main()
