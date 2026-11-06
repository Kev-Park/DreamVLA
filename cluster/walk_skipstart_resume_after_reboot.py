"""Finish a verified closed batch after a reboot, then resume the normal chain.

The operator supplies the current boot ID in the study's recovery configuration.
Existing raw recordings and completed conversion shards are reused unchanged.
No jobs are submitted to Adroit and no account-wide resource policy is changed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import gr00t_budget_chain as base

TAG = 'walk200_v3_skipstart_20261102'


def verify_shards(shard_root, sources, source_frames):
    """Check contiguous source coverage and each completed episode before merge."""
    shards, covered = [], []
    for inputs in sorted(shard_root.glob('in_*')):
        suffix = inputs.name.removeprefix('in_')
        shard = shard_root/'out'/('shard_'+suffix)
        links = sorted(inputs.glob('*.hdf5'))
        assert links and all(p.is_symlink() for p in links)
        targets = [str(p.resolve()) for p in links]
        info = base.read(shard/'meta/info.json')
        episodes = [json.loads(line) for line in
                    (shard/'meta/episodes.jsonl').read_text().splitlines() if line]
        episodes.sort(key=lambda row: row['episode_index'])
        assert info['total_episodes'] == len(episodes) == len(links)
        assert [row['episode_index'] for row in episodes] == list(range(len(links)))
        assert [row['length'] for row in episodes] == [source_frames[p] for p in targets]
        assert info['total_frames'] == sum(row['length'] for row in episodes)
        log = (shard_root/('log_'+suffix+'.txt')).read_text()
        assert re.search(r'^\[done\] wrote '+str(info['total_frames'])+r' frames', log, re.M)
        for row in episodes:
            index = row['episode_index']
            args = dict(episode_chunk=index//info['chunks_size'], episode_index=index)
            assert (shard/info['data_path'].format(**args)).stat().st_size > 0
            for key, feature in info['features'].items():
                if feature['dtype'] == 'video':
                    assert (shard/info['video_path'].format(video_key=key, **args)).stat().st_size > 0
        covered.extend(targets)
        shards.append(shard)
    assert shards and covered == list(map(str, sources)), 'Shard source coverage/order differs'
    return shards


def main():
    import h5py
    import numpy as np
    from walk_skipstart_chain import FreshChain

    chain = FreshChain()
    work = chain.work
    config = base.read(work/'post_reboot_recovery_config.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == config['boot_id'] and config['batch'] == 'r2_batch007'
    assert set(os.sched_getaffinity(0)) == {0, 3, 10, 13}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            arguments = (proc/'cmdline').read_bytes().split(bytes([0]))
            scripts = {Path(arg.decode(errors='replace')).name for arg in arguments[1:]}
            is_python = Path(arguments[0].decode(errors='replace')).name.startswith('python')
            if is_python:
                assert not scripts.intersection({
                    'walk_skipstart_chain.py', 'walk_skipstart_resume_after_reboot.py',
                    'walk_skipstart_resume_after_r2.py', 'walk_skipstart_recover_r2_batch005.py'})
                assert not ('collect_sonic_adapter.py' in scripts and
                            any(TAG.encode() in arg for arg in arguments))
        except FileNotFoundError:
            pass
    assert not (work/'abort.json').exists()
    (work/'controller.lock').write_text(str(os.getpid()))
    run = TAG+'_r1'
    trained = base.read(work/(run+'.trained.json'))
    assert trained['steps'] == 67542 and base.checkpoint_ok(Path(trained['checkpoint']))
    name = config['batch']
    report_path = work/(name+'.converted.json')
    recovery_path = work/'post_reboot_recovery_state.json'
    base.atomic(recovery_path, dict(pid=os.getpid(), status='validating_closed_batch',
                                   boot_id=boot, at=time.time()))
    if not report_path.exists():
        raw = chain.data/'raw'/name
        log = (work/(name+'.log')).read_bytes()
        assert hashlib.sha256(log).hexdigest() == config['collection_log_sha256']
        text = log.decode(errors='strict')
        base.verify_collection_logs([text], list(range(70, 80)))
        assert re.findall(r'Collection finished\..*errored=(\d+)', text) == ['0']
        sys.path.insert(0, str(chain.sim))
        from vla_sonic.grasp_success import score
        box = base.read(work/'box.json')
        rows, kept, source_frames = [], [], {}
        files = sorted(raw.rglob('*.hdf5'))
        assert len(files) == 10
        for path in files:
            with h5py.File(path, 'r', locking=False) as handle:
                metadata = json.loads(handle.attrs['metadata_json'])
                assert metadata['skip_start_frames'] == 0
                group = handle['data/demo_0']
                frames = len(group['actions'])
                assert frames > 0
                result = score(group['obs/object_pos'][()], group['obs/object_quat'][()],
                    group['obs/robot0_root_pos_w'][()], group['obs/robot0_root_quat_w'][()],
                    group['teleop/right_wrist'][()], box_lo=np.array(box['lo']), box_hi=np.array(box['hi']))
                rows.append(dict(path=str(path), frames=frames,
                    motion_id=int(metadata['motion_id']), passing=bool(result['success'])))
                if result['success']:
                    kept.append(path.resolve())
                    source_frames[str(path.resolve())] = frames
        assert sorted(row['motion_id'] for row in rows) == list(range(70, 80))
        dataset = chain.data/'converted'/name/'ds'
        assert not dataset.exists(), 'Existing merge needs separate reconciliation'
        shard_root = Path(config['shard_root'])
        assert shard_root.parent == dataset.parent and shard_root.name.startswith('ds.shards.')
        shards = verify_shards(shard_root, kept, source_frames)
        converter = base.ROOT/'GR00T-WholeBodyControl/.venv_data_collection/bin/python'
        chain.execute([converter, base.HERE/'merge_lerobot.py', '--shards', *shards,
                       '--out', dataset], 'convert_'+name+'_post_reboot_merge')
        info = base.read(dataset/'meta/info.json')
        assert info['total_episodes'] == len(kept)
        assert info['total_frames'] == sum(source_frames.values())
        manifest = []
        for path in sorted(raw.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                        digest.update(chunk)
                manifest.append(dict(path=str(path.relative_to(raw)), bytes=path.stat().st_size,
                                     sha256=digest.hexdigest()))
        base.atomic(report_path, dict(batch=name, requested_ids=list(range(70, 80)), rows=rows,
            retained=len(kept), frames=info['total_frames'], dataset=str(dataset),
            archive_source=str(raw), archive_files=manifest, skip_start_frames=0,
            seed=8, sampled=False, completed=time.time(),
            technical_recovery=dict(boot_id=boot, reused_completed_shards=True,
                                    repeated_rollouts=0, source_config=config)))
    else:
        report = base.read(report_path)
        assert report['requested_ids'] == list(range(70, 80))
        assert report['technical_recovery']['boot_id'] == boot
    base.atomic(recovery_path, dict(pid=os.getpid(), status='waiting_for_archive', at=time.time()))
    chain.await_archive(name)
    assert not (work/'abort.json').exists()
    base.atomic(recovery_path, dict(pid=os.getpid(), status='exec_normal_chain', at=time.time()))
    os.execv(sys.executable, [sys.executable, '-u', str(base.HERE/'walk_skipstart_chain.py')])


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        base.atomic(base.ROOT/'dline'/TAG/'post_reboot_recovery_state.json',
                    dict(pid=os.getpid(), status='stopped_for_review',
                         error=repr(exc), at=time.time()))
        raise
