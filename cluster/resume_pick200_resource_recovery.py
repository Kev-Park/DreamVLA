"""Resume the frozen pick200 experiment with corrected resource observations only.

Scientific code remains in the original 197a3c7 checkout. This entry point loads
that checkout explicitly; only GPU/CPU reservation functions come from this one.
The failed startup must be preserved and reconciled before normal execution.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SCIENCE_PIN = '197a3c7f9bce10604f4c8100c0d72d7549978377'
GROOT_PIN = '29c95b9cda2249b5285c84460c789d4fe3841d97'
TAG = 'pick200_original_spares57_v2'


def acquire_slots(chain, count, mb, allocator):
    """The frozen controller's lease logic, with a revalidated resource backend."""
    with chain.condition:
        while True:
            snapshot = None
            candidates = []
            for gpu, leases in chain.leases.items():
                candidates.extend([gpu] * max(0, (45000-sum(leases))//mb))
            required = 1 if count == 4 else count
            if len(candidates) >= required and (
                    count != 4 or len(candidates) >= count or len(chain.leases) >= 2):
                chosen = candidates[:count]
                for gpu in chosen:
                    chain.leases[gpu].append(mb)
                return chosen
            if len(chain.leases) < 2:
                snapshot = allocator.capacity()
                gpu = allocator.reserve(snapshot)
                if gpu is not None:
                    chain.leases[gpu] = []
                    continue
            if len(candidates) >= required:
                chosen = candidates[:count]
                for gpu in chosen:
                    chain.leases[gpu].append(mb)
                return chosen
            detail = (f' cpu={snapshot["cpu_fraction"]} free={snapshot["free"]}'
                      if snapshot is not None else '')
            chain.log(f'WAIT capacity count={count} memory={mb}MiB{detail}')
            chain.condition.wait(timeout=30)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = Path.home()/'kevin'
    frozen = root/'wt/pick200-original-spares'
    here = Path(__file__).resolve().parent
    for repo, pin in [(frozen, SCIENCE_PIN), (root/'Isaac-GR00T', GROOT_PIN)]:
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == pin
        for diff in [['diff', '--quiet'], ['diff', '--cached', '--quiet']]:
            subprocess.run(['git', '-C', str(repo), *diff], check=True)
    request = json.loads(args.request.read_text())
    assert request['run'] == TAG and request['launch_authorized'] is True
    work = root/'dline'/TAG
    assert args.request.resolve() == (work/'chain.request.json').resolve()
    data = root/'datasets'/f'{TAG}_r3/lerobot/ds'
    info = json.loads((data/'meta/info.json').read_text())
    assert (info['total_episodes'], info['total_frames']) == (707, 347512)
    assert not info.get('discarded_episode_indices')
    for i, steps in enumerate([35554, 65953, 95899]):
        trained = json.loads((work/f'{TAG}_r{i}.trained.json').read_text())
        assert trained['steps'] == steps and trained['batch_size'] == 16
    assert not (work/f'{TAG}_r3.trained.json').exists()

    # Resolve all scientific imports from the original immutable checkout first.
    sys.path.insert(0, str(frozen/'cluster'))
    spec = importlib.util.spec_from_file_location('pick200_recovery_capacity',
                                                here/'pick_original_bluesclues_queue.py')
    allocator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(allocator)
    from pick_spare_chain import SpareChain

    class RecoveryChain(SpareChain):
        def slots(self, count, mb):
            return acquire_slots(self, count, mb, allocator)

    evidence = dict(scientific_checkout=str(frozen), scientific_commit=SCIENCE_PIN,
                    groot_commit=GROOT_PIN, recovery_checkout=str(here.parent),
                    recovery_commit=subprocess.check_output(
                        ['git', '-C', str(here.parent), 'rev-parse', 'HEAD'], text=True).strip(),
                    allocator_sha256=hashlib.sha256((here/'pick_original_bluesclues_queue.py').read_bytes()).hexdigest(),
                    controller_script=Path(__file__).name, episodes=707, frames=347512,
                    target_steps=123802, global_batch_size=16, gradient_accumulation_steps=1,
                    scientific_configuration_changed=False)
    if args.preflight_only:
        print(json.dumps(evidence))
        return
    assert not (work/'abort.json').exists(), 'Preserve/reconcile the failed attempt first'
    assert not (work/f'ft_{TAG}_r3.log').exists(), 'Never overwrite the failed startup log'
    assert not (root/'checkpoints'/f'{TAG}_r3').exists(), 'Preserve failed output before fresh initialization'
    chain = RecoveryChain(request)
    evidence.update(controller_pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)))
    with (work/'spare_driver.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def publish(name, value):
            path = work/name
            temporary = path.with_suffix('.recovery.tmp')
            temporary.write_text(json.dumps(value, indent=2)+'\n')
            temporary.replace(path)
        publish('resource_recovery.json', evidence)
        try:
            chain.complete_spares(args.request)
            publish('chain.status.json', dict(pid=os.getpid(), phase='running', passing_originals=200,
                    controller_script=Path(__file__).name, resource_recovery=True))
            chain.run()
            publish('chain.status.json', dict(pid=os.getpid(), phase='done', passing_originals=200,
                    controller_script=Path(__file__).name, resource_recovery=True))
        except BaseException as error:
            publish('abort.json', dict(error=repr(error), pid=os.getpid()))
            publish('chain.status.json', dict(pid=os.getpid(), phase='needs_attention', error=repr(error),
                    controller_script=Path(__file__).name, resource_recovery=True))
            raise


if __name__ == '__main__':
    main()
