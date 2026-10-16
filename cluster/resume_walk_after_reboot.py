"""Operator recovery of the walking collections interrupted by the 2026-10-15 reboot.

Reuse finalized attempts, including valid no-grasp skips. Infrastructure-interrupted
attempts are recorded explicitly and reissued; completed attempts are never rerun.
Original logs and recordings remain untouched. No training recipe is changed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def progress(texts, expected):
    expected = list(expected)
    assert len(expected) == len(set(expected)), 'Duplicate selected references'
    started, written, skipped, verdicts = [], [], [], []
    for text in texts:
        started += list(map(int, re.findall(r'^\[INFO\] motion (\d+)_r0/\d+', text, re.M)))
        verdicts += list(map(int, re.findall(r'^\[rescue\] motion (\d+): pass A 4-rule', text, re.M)))
        written += list(map(int, re.findall(
            r'^\[rescue\] WROTE (?:student success|rescue) .*?__motion_(\d+)_r0\.hdf5', text, re.M)))
        skipped += list(map(int, re.findall(
            r'^\[rescue\] SKIP motion (\d+): no reference grasp frame', text, re.M)))
    for name, ids in [('starts', started), ('verdicts', verdicts), ('finalizations', written + skipped)]:
        assert len(ids) == len(set(ids)), f'Duplicate {name}'
        assert set(ids) <= set(expected), f'Unexpected {name}'
    assert set(verdicts) <= set(started)
    assert set(written + skipped) <= set(verdicts)
    done = set(written + skipped)
    return {'written_ids': sorted(written), 'valid_skip_ids': sorted(skipped),
            'completed_ids': sorted(done), 'remaining_ids': [i for i in expected if i not in done],
            'infrastructure_interrupted_ids': sorted(set(started) - done)}


def inspect_recordings(raw, written_ids):
    import h5py
    import numpy as np
    found = {}
    columns = ['actions', 'obs/object_pos', 'obs/object_quat', 'obs/robot0_root_pos_w',
               'obs/robot0_root_quat_w', 'obs/motion_token', 'teleop/right_wrist']
    for path in sorted(raw.rglob('*.hdf5')):
        match = re.fullmatch(r'sonic_adapter__Pick_sim2__motion_(\d+)_r0\.hdf5', path.name)
        assert match, f'Unrecognized recording requires review: {path}'
        motion = int(match[1])
        assert motion in written_ids and motion not in found, f'Unmatched/duplicate recording: {path}'
        with h5py.File(path, 'r', locking=False) as f:
            g = f['data/demo_0']; n = g['actions'].shape[0]
            assert n > 0 and all(g[name].shape[0] == n for name in columns), str(path)
            assert g['obs/motion_token'].shape == (n, 64), str(path)
            assert g['obs/ego_view_image'].shape == (n, 960, 1280, 3), str(path)
            assert all(np.isfinite(g[name][()]).all() for name in columns), str(path)
        found[motion] = {'path': str(path), 'bytes': path.stat().st_size, 'frames': n}
    assert set(found) == set(written_ids), 'Recorded outcomes and durable files differ'
    return {str(k): v for k, v in found.items()}


def build_plan(budget, boot_id):
    from gr00t_budget_chain import ROOT, read, atomic
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot_id
    boot = int(next(s.split()[1] for s in Path('/proc/stat').read_text().splitlines()
                    if s.startswith('btime ')))
    target = 3 if budget == 50 else 1
    work = ROOT / 'dline' / f'walk{budget}_v3_dagger57_20261013'
    recipe = read(work / 'recipe.json'); expected = recipe['motion_ids']
    assert recipe['budget'] == budget and len(expected) == budget
    assert recipe['batch_size'] == 16 and recipe['passes'] == 5.7
    assert recipe['learning_rate'] == 0.0001 and Path(recipe['expert']).name == 'model_4500.pt'
    assert not (work / f'collection_r{target}.json').exists()
    logs = sorted(work.glob(f'collect_r{target}_shard*.log'))
    assert len(logs) == (2 if budget == 50 else 4)
    checkpoints = set()
    for log in logs:
        process = read(log.with_suffix('.process.json'))
        assert process['started'] < boot, 'Worker was launched after the reboot'
        command = process['command']
        checkpoints.add(command[command.index('--dagger-vla-checkpoint') + 1])
    assert len(checkpoints) == 1, 'Interrupted workers used different students'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            assert not ('collect_sonic_adapter.py' in command and work.name in command), \
                'A collector for this chain is already live'
        except (FileNotFoundError, PermissionError):
            continue
    state = progress([p.read_text() for p in logs], expected)
    raw = ROOT / 'datasets' / f'{work.name}_collection_r{target}'
    recordings = inspect_recordings(raw, state['written_ids'])
    plan = dict(state, budget=budget, round=target, boot_id=boot_id, boot_time=boot,
                source_checkpoint=checkpoints.pop(),
                expected_ids=expected, raw=str(raw), checked_at=time.time(), recordings=recordings,
                original_logs=[{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                               for p in logs],
                label_prefix=f'collect_r{target}_reboot_20261015',
                scope='Basic data integrity and exact attempt coverage; no subjective behavior judgment')
    path = work / 'reboot_20261015_recovery_plan.json'
    if path.exists():
        previous = read(path)
        assert all(previous[k] == v for k, v in plan.items() if k != 'checked_at'), \
            'Existing recovery plan differs; reconcile before retrying'
        return previous
    atomic(path, plan)
    return plan


def recovered_chain(budget, plan):
    from walk_gr00t_chain import WalkChain
    import gr00t_budget_chain as base

    class Recovery(WalkChain):
        def collect(self, round_id, checkpoint):
            done = self.work / f'collection_r{round_id}.json'
            if round_id != plan['round'] or done.exists():
                return super().collect(round_id, checkpoint)
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id']
            assert self.ids == plan['expected_ids']
            assert str(checkpoint) == plan['source_checkpoint'], 'Recovery student differs'
            for row in plan['original_logs']:
                assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
            remaining = plan['remaining_ids']; out = Path(plan['raw'])
            self.log(f"RECOVER collection {round_id}: reuse {len(plan['completed_ids'])}, "
                     f"remaining {len(remaining)}, reboot-interrupted {plan['infrastructure_interrupted_ids']}")
            count = min(self.collection_workers, len(remaining))
            gpus = self.slots(count, 17000) if count else []
            labels = []
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, count)) as pool:
                    tasks = []
                    for k, gpu in enumerate(gpus):
                        idsfile = self.work / f"{plan['label_prefix']}_ids{k}.txt"
                        assert not idsfile.exists(), 'Existing recovery launch needs explicit reconciliation'
                        idsfile.write_text(' '.join(map(str, remaining[k::count])) + '\n')
                        label = f"{plan['label_prefix']}_shard{k}"; labels.append(label)
                        args = [sys.executable, self.sim / 'collect_sonic_adapter.py', '--headless',
                                '--task', 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0', '--waist-dof', '29',
                                '--sonic-pt', self.pt, '--encoder-mode', 'g1', '--residual-transform', 'additive_free',
                                '--residual-scale', '0.1', '--checkpoint-path', self.expert,
                                '--ref-motions-path', base.REFS, '--rollout-length', str(self.rollout_steps),
                                '--num-samples', '100000', '--seed', str(6 + round_id), '--output-directory', out,
                                '--dagger-vla-checkpoint', checkpoint, '--dagger-demo-root', self.work / 'demo_root',
                                '--dagger-motion-ids', idsfile, '--dagger-chunk', '8', '--dagger-rescue',
                                '--dagger-rescue-box', 'file:' + str(self.work / 'box.json'),
                                '--dagger-rescue-k-frac', '1.0', '--dagger-rescue-blend', '0',
                                '--dagger-expert-base', 'reference', '--dagger-rollouts', '1']
                        tasks.append(pool.submit(self.execute, args, label, gpu, self.training_cwd))
                    for task in tasks:
                        task.result()
                texts = [(self.work / (label + '.log')).read_text() for label in labels]
                base.verify_collection_logs(texts, remaining)
                resumed = progress(texts, remaining)
                assert not resumed['remaining_ids']
                combined = plan['written_ids'] + resumed['written_ids']
                inspect_recordings(out, combined)
                assert sorted(plan['completed_ids'] + resumed['completed_ids']) == sorted(self.ids)
                base.atomic(done, {'student_attempts': self.budget, 'motion_ids': sorted(self.ids),
                                  'errors': 0, 'checkpoint': str(checkpoint), 'output': str(out),
                                  'reboot_recovery': {'plan': 'reboot_20261015_recovery_plan.json',
                                      'reused_finalized_attempts': len(plan['completed_ids']),
                                      'new_finalized_attempts': len(remaining),
                                      'infrastructure_interrupted_ids': plan['infrastructure_interrupted_ids'],
                                      'new_logs': labels}})
            finally:
                for gpu in gpus:
                    self.release(gpu)
            return out

    chain = Recovery(budget)
    chain.collection_workers = 2 if budget == 50 else 4
    return chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, choices=[50, 200], required=True)
    parser.add_argument('--boot-id', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); plan = build_plan(args.budget, args.boot_id)
    print(json.dumps({k: v for k, v in plan.items() if k not in ['recordings', 'original_logs']}), flush=True)
    if not args.preflight_only:
        recovered_chain(args.budget, plan).run()


if __name__ == '__main__':
    main()
