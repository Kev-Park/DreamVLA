"""Fresh 200-demo walking chain with historical skip-start restored.

Raw batches wait for a checksum-verified local archive acknowledgement before
continuing. Existing inputs and checkpoints are never reused as training data.
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time

import gr00t_budget_chain as base
from walk_gr00t_chain import WalkChain
from watch_t20max_v3 import command, environment

TAG = 'walk200_v3_skipstart_20261102'


def wait_owned_controller(pid, poll_seconds=30):
    """Wait for this exact controller, including Python builds without pidfd."""
    import select
    proc = Path('/proc')/str(pid)
    assert proc.stat().st_uid == os.getuid()
    assert b'walk_skipstart_chain.py' in (proc/'cmdline').read_bytes()
    if hasattr(os, 'pidfd_open'):
        fd = os.pidfd_open(pid)
        try:
            select.select([fd], [], [])
        finally:
            os.close(fd)
        return
    # /proc start time prevents a reused PID from being mistaken for the
    # original process. The caller separately requires its completion receipt.
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    started = fields[19]
    while True:
        try:
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            return
        if fields[19] != started or fields[0] == 'Z':
            return
        time.sleep(poll_seconds)


class FreshChain(WalkChain):
    def __init__(self):
        # Avoid creating the old chain's working directory or changing it.
        base.Chain.__init__(self, 200, tag=TAG)
        self.expert = base.ROOT/'residuals/T20_max_v3_model_4500.pt'
        self.rollout_steps = 1200
        self.eval_workers = 1
        self.collection_workers = 1
        self.sim_task_threads = 4
        self.training_cwd = base.HERE.parent/'Training'
        self.score_script = base.HERE/'walk_gr00t_score.py'
        self.env = environment(0)
        self.env.update(PYTHONPATH=str(self.sim), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        OPENBLAS_NUM_THREADS='1', PXR_WORK_THREAD_LIMIT='4',
                        TOKENIZERS_PARALLELISM='false')
        self.data = base.ROOT/'datasets'/TAG
        self.data.mkdir(exist_ok=True)
        self.ids = list(range(200))
        self.sources = []
        self.collection_gpu = 0

    def slots(self, count, mb):
        assert count == 1
        # Explicit operator-reviewed devices; do not spread onto other GPUs.
        gpu = 5 if threading.current_thread() is not threading.main_thread() else 4
        return self.reserve(gpu, mb)

    def reserve(self, gpu, mb):
        while True:
            out = subprocess.check_output([sys.executable, str(base.HERE/'gpu_slots.py'),
                'reserve', '--owner', str(os.getpid()), '--count', '1', '--mb', str(mb),
                '--gpu', str(gpu)], text=True)
            ids = list(map(int, out.split()))
            if ids:
                assert ids == [gpu]
                return ids
            self.log(f'WAIT reviewed GPU {gpu}')
            time.sleep(30)

    def execute(self, args, label, gpu=None, cwd=None):
        args = list(args)
        if any(Path(str(a)).name == 'collect_sonic_adapter.py' for a in args):
            assert '--skip-start-frames' not in args
            args += ['--skip-start-frames', '0']
        # Evaluation deliberately preserves the existing frame-0 protocol.
        return super().execute(args, label, gpu, cwd)

    def batch(self, name, ids, sampled=False, seed=0, checkpoint=None):
        import h5py
        import numpy as np
        sys.path.insert(0, str(self.sim))
        from vla_sonic.grasp_success import score
        report = self.work/(name+'.converted.json')
        if report.exists():
            result = base.read(report)
            assert result['requested_ids'] == ids
            self.await_archive(name)
            return result
        reserve = getattr(self, 'collection_reserve_bytes',
                          65_000_000_000 if self.collection_gpu else 25_000_000_000)
        while shutil.disk_usage(self.data).free < reserve:
            self.log(f'WAIT disk: need{reserve}bytes before next bounded batch; no deletion')
            time.sleep(30)
        raw = self.data/'raw'/name
        raw.mkdir(parents=True, exist_ok=False)
        idsfile = self.work/(name+'.ids.txt')
        idsfile.write_text(' '.join(map(str, ids))+'\n')
        args = command('collect_sonic_adapter.py', 'StitchMAX200v3')
        args[args.index('--task')+1] = 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0'
        args += ['--checkpoint-path', str(self.expert), '--rollout-length', '1200',
                 '--seed', str(seed), '--output-directory', str(raw)]
        if checkpoint is None:
            args += ['--num-samples', str(len(ids)), '--motion-ids', str(idsfile)]
            if sampled:
                args.append('--sample-expert-actions')
        else:
            args += ['--num-samples', '100000', '--dagger-vla-checkpoint', str(checkpoint),
                     '--dagger-demo-root', str(self.work/'demo_root'), '--dagger-motion-ids', str(idsfile),
                     '--dagger-chunk', '8', '--dagger-rescue', '--dagger-rescue-box', 'file:'+str(self.work/'box.json'),
                     '--dagger-rescue-k-frac', '1.0', '--dagger-rescue-blend', '0',
                     '--dagger-expert-base', 'reference', '--dagger-rollouts', '1']
        gpu = self.reserve(self.collection_gpu, 20000)[0]
        try:
            self.execute(args, name, gpu, self.training_cwd)
        finally:
            self.release(gpu)
        text = (self.work/(name+'.log')).read_text()
        assert re.findall(r'Collection finished\..*errored=(\d+)', text) == ['0']
        if checkpoint:
            base.verify_collection_logs([text], ids)
        else:
            attempted = list(map(int, re.findall(r'\[INFO\] motion (\d+)(?:/|_)', text)))
            assert sorted(attempted) == sorted(ids), (attempted, ids)
        box = base.read(self.work/'box.json')
        rows, kept = [], []
        for p in sorted(raw.rglob('*.hdf5')):
            with h5py.File(p, 'r', locking=False) as h:
                g = h['data/demo_0']
                md = json.loads(h.attrs['metadata_json'])
                assert md['skip_start_frames'] == 0
                n = len(g['actions'])
                result = score(g['obs/object_pos'][()], g['obs/object_quat'][()],
                    g['obs/robot0_root_pos_w'][()], g['obs/robot0_root_quat_w'][()],
                    g['teleop/right_wrist'][()], box_lo=np.array(box['lo']), box_hi=np.array(box['hi']))
                rows.append(dict(path=str(p), frames=n, motion_id=int(md['motion_id']), passing=bool(result['success'])))
                if result['success']:
                    kept.append(p)
        assert len(rows) <= len(ids) and len({r['motion_id'] for r in rows}) == len(rows)
        dataset = self.data/'converted'/name/'ds'
        if kept:
            inputs = self.data/'conversion_inputs'/name
            inputs.mkdir(parents=True, exist_ok=False)
            for i, p in enumerate(kept):
                (inputs/f'{i:05d}__{p.name}').symlink_to(p.resolve())
            self.execute(['bash', base.HERE/'convert_sharded.sh', '--hdf5-root', inputs,
                '--recursive', '--output-path', dataset.parent, '--dataset-name', 'ds',
                '--task-prompt', 'pick up the mustard bottle', '--fps', '50', '--shards', '4'],
                'convert_'+name)
            info = base.read(dataset/'meta/info.json')
            assert info['total_episodes'] == len(kept)
        else:
            info = {'total_episodes': 0, 'total_frames': 0}
        manifest = []
        for p in sorted(raw.rglob('*')):
            if p.is_file():
                digest = hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda: f.read(8*1024*1024), b''):
                        digest.update(chunk)
                manifest.append(dict(path=str(p.relative_to(raw)), bytes=p.stat().st_size, sha256=digest.hexdigest()))
        result = dict(batch=name, requested_ids=ids, rows=rows, retained=len(kept),
            dataset=str(dataset) if kept else None, frames=info['total_frames'],
            archive_source=str(raw), archive_files=manifest, skip_start_frames=0,
            seed=seed, sampled=sampled, completed=time.time())
        base.atomic(report, result)
        self.await_archive(name)
        return result

    def await_archive(self, name):
        receipt = self.work/(name+'.archived.json')
        while not receipt.exists():
            base.atomic(self.work/'archive_wait.json', dict(batch=name, at=time.time(), pid=os.getpid()))
            time.sleep(30)
        assert base.read(receipt)['checksum_verified'] is True

    def merge(self, name, datasets, expected):
        out = self.data/name/'lerobot/ds'
        done = self.work/(name+'.dataset.json')
        if not done.exists():
            assert not out.exists(), 'Incomplete dataset requires reconciliation'
            # Use the exact environment convert_sharded.sh uses for its merges;
            # the simulation interpreter intentionally has no LeRobot package.
            wbc = Path(self.env.get('GR00T_WHOLEBODYCONTROL_DIR',
                                    str(base.ROOT/'GR00T-WholeBodyControl')))
            self.execute([wbc/'.venv_data_collection/bin/python', base.HERE/'merge_lerobot.py', '--shards', *datasets,
                          '--out', out], 'merge_'+name)
            info = base.read(out/'meta/info.json')
            assert info['total_episodes'] == expected
            base.atomic(done, dict(dataset=str(out), episodes=expected, frames=info['total_frames']))
        assert base.read(out/'meta/info.json')['total_episodes'] == expected
        return out

    def prepare(self):
        # Preserve the most recent expert's fixed scorer/closed-pose inputs.
        source = base.ROOT/'dline/walk200_v3_histwindow65_20261029/inputs'
        if not (self.work/'box.json').exists():
            shutil.copy2(source/'box.json', self.work/'box.json')
        assert base.read(self.work/'box.json') == base.read(source/'box.json')
        cfg = base.read(source/'config.json')
        # Input preparation copies the same small closed-pose demo files here.
        assert list((self.work/'demo_root').glob('*.hdf5'))
        order = list(range(200)); random.Random(42).shuffle(order)
        datasets, selected = [], []
        def sweep(ids, prefix, sampled, seed):
            for k in range(0, len(ids), 10):
                r = self.batch(f'{prefix}_{k//10:03d}', ids[k:k+10], sampled, seed)
                if r['retained']:
                    datasets.append(r['dataset'])
                    selected.extend(x for x in r['rows'] if x['passing'])
        sweep(order, 'base_initial', False, 0)
        for attempt in range(1, 4):
            missing = sorted(set(order)-{r['motion_id'] for r in selected})[:200-len(selected)]
            if not missing:
                break
            sweep(missing, f'base_retry{attempt}', True, 12000+attempt)
        fill = 0
        while len(selected) < 200:
            fill += 1
            assert fill <= 30 and selected
            ids = sorted({r['motion_id'] for r in selected})[:200-len(selected)]
            sweep(ids, f'base_fill{fill}', True, 120000+fill)
        assert len(selected) == 200
        base.atomic(self.work/'base_manifest.json', dict(episodes=selected, datasets=datasets,
            count=200, distinct_references=len({r['motion_id'] for r in selected}),
            original_reference_order=[Path(p).name for p in glob.glob(str(base.ROOT/'ref_motions/StitchMAX200v3/*.pkl'))]))
        return self.merge('r0', datasets, 200)

    def collect(self, round_id, checkpoint):
        results = [self.batch(f'r{round_id}_batch{k//10:03d}', self.ids[k:k+10],
                    seed=6+round_id, checkpoint=checkpoint) for k in range(0, 200, 10)]
        assert sum(len(r['requested_ids']) for r in results) == 200
        base.atomic(self.work/f'collection_r{round_id}.json', dict(batches=results, student_attempts=200))
        return results

    def dataset(self, round_id, results, prior):
        shards = [str(prior)]+[r['dataset'] for r in results if r['retained']]
        count = base.read(prior/'meta/info.json')['total_episodes']+sum(r['retained'] for r in results)
        return self.merge(f'r{round_id}', shards, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-peer-batches', nargs='+', type=int, choices=range(4,20))
    parser.add_argument('--collection-gpu', type=int, choices=(4,5,6,7))
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('--dagger-peer-round', type=int, choices=(1,2,3))
    parser.add_argument('--dagger-peer-batches', nargs='+', type=int, choices=range(5,20))
    args = parser.parse_args()
    assert not (args.base_peer_batches and args.dagger_peer_batches)
    assert bool(args.base_peer_batches or args.dagger_peer_batches) == (args.collection_gpu is not None)
    assert bool(args.dagger_peer_batches) == (args.dagger_peer_round is not None)
    if args.after_pid is not None:
        assert args.base_peer_batches and args.collection_gpu in (5,6,7)
        deferred = base.ROOT/'dline'/TAG/f'deferred_base_gpu{args.collection_gpu}.json'
        base.atomic(deferred, dict(status='waiting', pid=os.getpid(),
            predecessor_pid=args.after_pid, batches=args.base_peer_batches, at=time.time()))
        try:
            wait_owned_controller(args.after_pid)
            receipt = base.read(base.ROOT/'dline'/TAG/f'base_peer_gpu{args.collection_gpu}.json')
            assert receipt['pid'] == args.after_pid and receipt['status'] == 'complete', receipt
            base.atomic(deferred, dict(status='ready', pid=os.getpid(),
                predecessor_pid=args.after_pid, batches=args.base_peer_batches, at=time.time()))
        except BaseException as exc:
            base.atomic(deferred, dict(status='failed', pid=os.getpid(),
                predecessor_pid=args.after_pid, error=repr(exc), at=time.time()))
            raise
    chain = FreshChain()
    if args.dagger_peer_round is not None:
        round_id = args.dagger_peer_round
        assert args.collection_gpu in (0,4,6,7,9)
        assert len(set(args.dagger_peer_batches)) == len(args.dagger_peer_batches)
        trained = base.read(chain.work/f'{TAG}_r{round_id-1}.trained.json')
        checkpoint = Path(trained['checkpoint'])
        assert base.checkpoint_ok(checkpoint)
        chain.collection_gpu = args.collection_gpu
        # A serial peer may reuse GPU 0 after the main lane releases it.
        # Keep the peer's existing disk reserve on every supported device.
        chain.collection_reserve_bytes = 45_000_000_000
        state = chain.work/f'dagger_peer_r{round_id}_gpu{args.collection_gpu}.json'
        try:
            for batch in args.dagger_peer_batches:
                base.atomic(state,dict(pid=os.getpid(),batch=batch,round=round_id,
                    checkpoint=str(checkpoint),at=time.time(),status='collecting'))
                chain.batch(f'r{round_id}_batch{batch:03d}',chain.ids[batch*10:(batch+1)*10],
                            seed=6+round_id,checkpoint=checkpoint)
            base.atomic(state,dict(pid=os.getpid(),round=round_id,at=time.time(),status='complete'))
        except BaseException as exc:
            base.atomic(state,dict(pid=os.getpid(),round=round_id,at=time.time(),status='failed',error=repr(exc)))
            raise
        return
    if args.base_peer_batches:
        # Parent owns the initial batches. Disjoint future batches must be
        # completed before the parent arrives; never replace an active worker.
        assert len(set(args.base_peer_batches)) == len(args.base_peer_batches)
        chain.collection_gpu = args.collection_gpu
        order = list(range(200)); random.Random(42).shuffle(order)
        state = chain.work/f'base_peer_gpu{args.collection_gpu}.json'
        try:
            for batch in args.base_peer_batches:
                base.atomic(state, dict(pid=os.getpid(), batch=batch, at=time.time(), status='collecting'))
                chain.batch(f'base_initial_{batch:03d}', order[batch*10:(batch+1)*10])
            base.atomic(state, dict(pid=os.getpid(), at=time.time(), status='complete'))
            if args.after_pid is not None:
                base.atomic(deferred, dict(status='complete', pid=os.getpid(),
                    predecessor_pid=args.after_pid, batches=args.base_peer_batches, at=time.time()))
        except BaseException as exc:
            base.atomic(state, dict(pid=os.getpid(), at=time.time(), status='failed', error=repr(exc)))
            if args.after_pid is not None:
                base.atomic(deferred, dict(status='failed', pid=os.getpid(),
                    predecessor_pid=args.after_pid, error=repr(exc), at=time.time()))
            raise
        return
    try:
        chain.run()
    except BaseException as exc:
        base.atomic(chain.work/'abort.json', dict(at=time.time(), pid=os.getpid(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
