"""Verify all four collection groups, convert passes and append the R2 prefix.

Run with the reviewed .venv_data_collection Python inside Isaac on Adroit.
No jobs are submitted, no recordings are deleted, and no normalization caches
are inherited. Partial outputs are preserved for operator diagnosis.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from walk200_round3_inputs import BOX_SHA, EPISODES, FRAMES, PREFIX, TAG, digest, student_inputs


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def group_coverage(work):
    """Require each attempt to be scored and finalized once, with no retries."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gr00t_budget_chain import verify_collection_logs
    manifest_path = work / 'adroit_round3_inputs.ready.json'
    manifest_sha = digest(manifest_path)
    checkpoint_suffix = Path(read(manifest_path)['student_checkpoint_relative']).parts[-3:]
    jobs, written, skipped = [], [], []
    for group in range(4):
        directory = work / 'adroit_round3_groups' / str(group)
        assert not (directory / 'group.abort.json').exists(), 'Collection group aborted'
        launch, done = read(directory / 'group.launch.json'), read(directory / 'group.done.json')
        expected = list(range(group * 50, (group + 1) * 50))
        assert launch['group'] == done['group'] == group
        assert launch['motion_ids'] == done['motion_ids'] == expected
        assert launch['source_manifest_sha256'] == manifest_sha
        assert done['student_attempts'] == 50 and done['errors'] == 0
        assert launch['checkpoint'] == done['checkpoint']
        assert Path(launch['checkpoint']).parts[-3:] == checkpoint_suffix
        assert launch['slurm_job_id'] == done['slurm_job_id']
        job = str(done['slurm_job_id'])
        assert job.isdigit() and job not in jobs
        jobs.append(job)
        texts = [(directory / f'collect_r3_shard{k}.log').read_text() for k in range(2)]
        verify_collection_logs(texts, expected)
        finalized = []
        for text in texts:
            saved = list(map(int, re.findall(
                r'^\[rescue\] WROTE (?:student success|rescue) .*?__motion_(\d+)_r0\.hdf5', text, re.M)))
            omitted = list(map(int, re.findall(
                r'^\[rescue\] SKIP motion (\d+): no reference grasp frame', text, re.M)))
            written.extend(saved)
            skipped.extend(omitted)
            finalized.extend(saved + omitted)
        assert sorted(finalized) == expected, 'Missing or duplicate final recording disposition'
    assert sorted(written + skipped) == list(range(200))
    return {'slurm_jobs': jobs, 'written_ids': sorted(written), 'skipped_ids': sorted(skipped)}


def completed_slurm_jobs(jobs):
    text = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-n', '-X', '-P',
                                    '-o', 'JobIDRaw,State,ExitCode'], text=True, timeout=30)
    rows = [row.strip().split('|') for row in text.splitlines() if row.strip()]
    assert len(rows) == len(jobs) and {row[0] for row in rows} == set(jobs)
    assert all(row[1:3] == ['COMPLETED', '0:0'] for row in rows), rows
    return rows


def main():
    root = (Path.home() / 'kevin').resolve(strict=True)
    work = root / 'dline' / TAG
    wbc = root / 'GR00T-WholeBodyControl'
    assert Path(sys.prefix).resolve() == (wbc / '.venv_data_collection').resolve(strict=True)
    checkpoint, _, _ = student_inputs(root)
    coverage = group_coverage(work)
    slurm = completed_slurm_jobs(coverage['slurm_jobs'])
    raw = root / 'datasets' / (TAG + '_collection_r3')
    raw_files = sorted(raw.rglob('*.hdf5'))
    recorded = []
    for path in raw_files:
        match = re.fullmatch(r'sonic_adapter__Pick_sim2__motion_(\d+)_r0\.hdf5', path.name)
        assert match and path.resolve(strict=True).is_relative_to(root), str(path)
        recorded.append(int(match[1]))
    assert sorted(recorded) == coverage['written_ids'], 'Raw files disagree with finalized collection logs'
    catalog_path = work / 'adroit_round2_raw_provenance.source.json'
    catalog = read(catalog_path)
    assert catalog['source_host'] == 'bluesclues'
    assert catalog['episodes'] == EPISODES and catalog['frames'] == FRAMES
    assert [row['episode_index'] for row in catalog['entries']] == list(range(EPISODES))
    assert sum(row['frames'] for row in catalog['entries']) == FRAMES
    box_path = work / 'box.json'
    assert digest(box_path) == BOX_SHA
    box = read(box_path)
    # Use the original Chain.dataset filter, with the same inputs and box.
    import h5py
    import numpy as np
    here = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here.parent / 'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.grasp_success import score
    scored, kept, added_frames = [], [], 0
    for path in raw_files:
        before = path.stat()
        with h5py.File(path, 'r', locking=False) as f:
            g = f['data/demo_0']
            frames = len(g['obs/object_pos'])
            assert frames > 0
            ok = bool(score(g['obs/object_pos'][()], g['obs/object_quat'][()],
                            g['obs/robot0_root_pos_w'][()], g['obs/robot0_root_quat_w'][()],
                            g['teleop/right_wrist'][()], box_lo=np.array(box['lo']),
                            box_hi=np.array(box['hi']))['success'])
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        scored.append({'source_host': 'adroit', 'source_relative': path.relative_to(root).as_posix(),
                       'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
                       'frames': frames, 'four_rule_pass': ok})
        if ok:
            kept.append(path)
            added_frames += frames
    assert len(kept) <= 200
    destination = root / 'datasets' / (TAG + '_r3')
    assert not destination.exists(), 'Reconcile previous assembly; never overwrite partial output'
    destination.mkdir()
    launch = {'at': time.time(), 'pid': os.getpid(), 'checkpoint': str(checkpoint),
              'coverage': coverage, 'slurm_accounting': slurm, 'recordings': scored,
              'prefix_provenance_sha256': digest(catalog_path), 'prefix_dataset': str(root / PREFIX),
              'prefix_episodes': EPISODES, 'prefix_frames': FRAMES,
              'retained_new_episodes': len(kept), 'new_frames': added_frames,
              'raw_files_deleted': 0, 'compression': False}
    write_new(destination / 'assembly.launch.json', launch)
    shard_roots = []
    jobs = []
    shards = min(4, len(kept))
    for shard in range(shards):
        lo, hi = round(shard * len(kept) / shards), round((shard + 1) * len(kept) / shards)
        inputs = destination / 'delta_hdf5' / f'shard_{shard}'
        inputs.mkdir(parents=True)
        for index in range(lo, hi):
            (inputs / f'{index:06d}__{kept[index].name}').symlink_to(kept[index].resolve(strict=True))
        name = f'delta_{shard}'
        shard_roots.append(destination / 'converted_shards' / name)
        jobs.append((shard, [sys.executable, str(wbc / 'gear_sonic/scripts/convert_isaac_hdf5_to_lerobot.py'),
                            '--hdf5-root', str(inputs), '--output-path', str(destination / 'converted_shards'),
                            '--dataset-name', name, '--task-prompt', 'pick up the mustard bottle', '--fps', '50']))

    def convert(job):
        shard, command = job
        with (destination / f'convert_delta_{shard}.log').open('x') as log:
            subprocess.run(command, cwd=wbc / 'gear_sonic/scripts', stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    if jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(pool.map(convert, jobs))
    merged = destination / 'lerobot/ds'
    subprocess.run([sys.executable, str(here / 'merge_lerobot.py'), '--shards',
                    str(root / PREFIX), *map(str, shard_roots), '--out', str(merged)], check=True)
    info = read(merged / 'meta/info.json')
    assert info['total_episodes'] == EPISODES + len(kept)
    assert info['total_frames'] == FRAMES + added_frames
    assert not info.get('discarded_episode_indices')
    assert all(not (merged / 'meta' / name).exists() for name in ('stats.json', 'relative_stats.json'))
    episodes = [json.loads(line) for line in (merged / 'meta/episodes.jsonl').read_text().splitlines()]
    assert [row['episode_index'] for row in episodes] == list(range(info['total_episodes']))
    assert sum(row['length'] for row in episodes) == info['total_frames']
    # Verify the reused prefix's values and encoded videos after the merge.
    import av
    import pyarrow.parquet as pq
    for index, episode in enumerate(episodes):
        args = {'episode_chunk': index // info['chunks_size'], 'episode_index': index}
        relative = info['data_path'].format(**args)
        assert pq.read_metadata(merged / relative).num_rows == episode['length']
        if index < EPISODES:
            assert pq.read_table(root / PREFIX / relative).equals(pq.read_table(merged / relative))
        for key, feature in info['features'].items():
            if feature['dtype'] == 'video':
                video = info['video_path'].format(video_key=key, **args)
                if index < EPISODES:
                    assert digest(root / PREFIX / video) == digest(merged / video)
                with av.open(str(merged / video)) as container:
                    assert len(container.streams.video) == 1
                    stream = container.streams.video[0]
                    assert stream.frames == episode['length']
                    assert float(stream.average_rate) == info['fps']
                    assert [stream.height, stream.width] == feature['shape'][:2]
    for row in scored:
        stat = (root / row['source_relative']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns'])
    new_rows = [dict(row, episode_index=EPISODES + index)
                for index, row in enumerate(row for row in scored if row['four_rule_pass'])]
    write_new(destination / 'source_episodes.json', {
        'prefix_provenance': str(catalog_path), 'prefix_sha256': digest(catalog_path),
        'prefix_raw_host': 'bluesclues', 'new_episodes': new_rows})
    result = {'episodes': info['total_episodes'], 'frames': info['total_frames'],
              'new_passes': len(kept), 'new_attempts': 200,
              'steps': (info['total_frames'] * 57 + 159) // 160,
              'batch_size': 16, 'effective_passes_target': 5.7,
              'prefix_values_and_video_bytes_verified': True,
              'all_episode_rows_and_video_headers_verified': True,
              'normalization': 'Generate fresh global stats at fine-tune startup',
              'raw_files_deleted': 0, 'shard_outputs_preserved': True}
    write_new(destination / 'built.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
