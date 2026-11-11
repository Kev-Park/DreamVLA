"""Recover one confirmed pre-startup Kit PID collision, then resume at a safe boundary.

Existing eval/collection/training workers are never stopped. The failed worker's
reservation changes ownership without increasing reserved memory. All failed
logs remain archived; completion is recorded only after all 100 evals are scored.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time

from gr00t_budget_chain import atomic, read, shared_memory_pid_collision
from walk_gr00t_chain import WalkChain


def live(pid):
    return Path(f'/proc/{pid}').exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, choices=[10, 50, 200], required=True)
    parser.add_argument('--round', type=int, choices=range(4), required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    args = parser.parse_args()
    chain = WalkChain(args.budget)
    work = chain.work
    lock = (work/'startup_repair.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = f'{chain.tag}_r{args.round}'
    label = f'eval_{run}_shard{args.shard}'
    handle = read(work/(label+'.process.json'))
    failed_pid, gpu = handle['pid'], handle['gpu']
    parent = args.controller_pid
    assert read(work/'controller.launch.json')['pid'] == parent
    parent_cmd = Path(f'/proc/{parent}/cmdline').read_text().replace('\0', ' ')
    assert 'walk_gr00t_chain.py' in parent_cmd and f'--budget {args.budget}' in parent_cmd
    trained = read(work/(run+'.trained.json'))
    command = handle['command']
    assert command[command.index('--vla-checkpoint')+1] == trained['checkpoint']
    assert command[command.index('--ref-motions-path')+1] == 'StitchMAX100v3_eval'
    assert command[command.index('--num-episodes')+1] == '25'
    assert command[command.index('--seed')+1] == str(100+args.shard)
    output = Path(command[command.index('--traj-dump')+1])
    assert output == chain.work.parents[1]/'eval_videos'/run/f'b{args.shard}'
    assert not list(output.parent.glob(output.name+'_ep*_traj.npz'))
    failed_text = (work/(label+'.log')).read_text()
    assert not live(failed_pid) and shared_memory_pid_collision(failed_text, failed_pid)
    assert not (work/(run+'.eval.json')).exists()
    receipt = work/f'{run}.startup_repair.json'
    state = {'time': time.time(), 'pid': os.getpid(), 'controller_pid': parent,
             'failed_pid': failed_pid, 'shard': args.shard, 'run': run,
             'status': 'validated_before_startup', 'no_active_jobs_stopped': True}

    def heartbeat(status):
        state.update(status=status, heartbeat=time.time())
        atomic(receipt, state)

    # The old evaluator still holds all four reservations until its three good
    # siblings finish. Transfer exactly the failed worker's slot, under the
    # allocator's own lock, so this replacement remains accounted afterwards.
    root = chain.work.parent
    with (root/'gpu_slots.lock').open('a+') as slots_lock:
        fcntl.flock(slots_lock, fcntl.LOCK_EX)
        entries = read(root/'gpu_slots.json')
        assert live(parent) and not live(failed_pid)
        index = next(i for i, e in enumerate(entries)
                     if e == {'owner': parent, 'gpu': gpu, 'mb': 17000})
        state['reservations_before'] = entries.copy()
        entries[index] = {'owner': os.getpid(), 'gpu': gpu, 'mb': 17000}
        atomic(root/'gpu_slots.json', entries)
        state['reservations_after'] = entries
        heartbeat('replacement_slot_transferred')
    try:
        archive = work/'startup_collisions'/f'{label}_{failed_pid}'
        archive.mkdir(parents=True, exist_ok=False)
        for suffix in ('.log', '.process.json', '.heartbeat.json'):
            path = work/(label+suffix)
            if path.exists():
                path.rename(archive/path.name)
        state['archive'] = str(archive)
        heartbeat('rerunning_failed_shard')
        chain.execute(command, label, gpu, chain.training_cwd)
    finally:
        chain.release(gpu)

    heartbeat('waiting_for_existing_eval_siblings')
    while True:
        handles = [read(work/f'eval_{run}_shard{k}.process.json') for k in range(4)]
        if not any(live(h['pid']) for h in handles):
            break
        heartbeat('waiting_for_existing_eval_siblings')
        time.sleep(30)
    status = (work/'status').read_text()
    for k in range(4):
        sibling = f'eval_{run}_shard{k}'
        exits = re.findall(r'EXIT '+re.escape(sibling)+r' code=(-?\d+)', status)
        assert exits and exits[-1] == '0', (sibling, exits)
        text = (work/(sibling+'.log')).read_text(errors='replace')
        assert 'VLA + SONIC EVAL SUMMARY' in text
        assert re.search(r'Completed episodes:\s+25\b', text)
        assert trained['checkpoint'] in text
    heartbeat('scoring_verified_100_reference_eval')
    chain.execute([command[2], chain.score_script, output.parent, work/'box.json'], f'score_{run}')
    report = read(output.parent/'results.json')
    assert report['pool'] == 'StitchMAX100v3_eval' and report['episodes'] == 100
    assert sorted(r['motion_id'] for r in report['per_episode']) == list(range(100))
    atomic(work/(run+'.eval.json'), report)
    heartbeat('eval_complete_waiting_for_original_controller_boundary')

    # Its original failed Future remains failed: allow collection and conversion
    # to finish naturally, then resume from their verified completion manifests.
    while live(parent):
        heartbeat('eval_complete_waiting_for_original_controller_boundary')
        time.sleep(30)
    failure = read(work/'abort.json')
    assert failure['pid'] == parent and label+' exited -6' in failure['error'], failure
    assert (work/f'collection_r{args.round+1}.json').exists()
    assert (chain.work.parents[1]/'datasets'/f'{chain.tag}_r{args.round+1}'/'built.json').exists()
    for path in work.glob('*.process.json'):
        assert not live(read(path)['pid']), str(path)
    (work/'abort.json').rename(archive/'controller_abort.json')
    prior = read(work/'controller.launch.json')
    atomic(archive/'controller.launch.json', prior)
    atomic(work/'controller.launch.json', dict(prior, pid=os.getpid(), started=time.time(),
           recovery_script=str(Path(__file__).resolve()), recovered_from_pid=parent))
    heartbeat('resuming_from_verified_completed_stages')
    try:
        chain.run()
    except BaseException as exc:
        atomic(work/'abort.json', {'time': time.time(), 'error': repr(exc), 'pid': os.getpid()})
        heartbeat('failed_for_review')
        raise
    heartbeat('complete')


if __name__ == '__main__':
    main()
