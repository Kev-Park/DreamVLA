"""Reuse the 173 verified original demos without decoding/re-encoding videos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    a = parser.parse_args()
    request = json.loads(a.request.read_text())
    assert request['launch_authorized'] is True
    rows = request['rows']
    assert len(rows) == len({r['source_motion_id'] for r in rows}) == 173
    assert all(r['passing'] and r['reference_file'] == f"pick_{r['source_motion_id']}.pkl" for r in rows)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    root = Path.home() / 'kevin'
    original = root / 'datasets/af60v8f_base'
    dataset = original / 'lerobot/ds'
    sources = sorted((original / 'hdf5').glob('*.hdf5'))
    info = json.loads((dataset / 'meta/info.json').read_text())
    assert info['total_episodes'] == len(sources) == 342
    prefix = root / 'datasets/pick200_original_base_prefix'
    assert not prefix.exists(), 'Partial preparation needs reconciliation; never overwrite it'
    inputs = prefix / 'hdf5'
    inputs.mkdir(parents=True)
    for i, row in enumerate(rows):
        source = Path(row['source']).resolve(strict=True)
        assert sources[row['converted_episode_index']].resolve(strict=True) == source
        (inputs / f'{i:05d}__{source.name}').symlink_to(source)
    indices = prefix / 'episode_indices.json'
    indices.write_text(json.dumps([r['converted_episode_index'] for r in rows])+'\n')
    target = prefix / 'lerobot/ds'
    subprocess.run([sys.executable, str(Path(__file__).with_name('merge_lerobot.py')),
                    '--shards', str(dataset), '--out', str(target), '--episode-indices', str(indices)], check=True)
    import pyarrow.parquet as pq
    new_info = json.loads((target / 'meta/info.json').read_text())
    assert new_info['total_episodes'] == 173 and new_info['total_frames'] == sum(r['frames'] for r in rows)
    proofs = []
    for i, row in enumerate(rows):
        old_index = row['converted_episode_index']
        old_path = dataset / info['data_path'].format(episode_chunk=old_index//info['chunks_size'], episode_index=old_index)
        new_path = target / new_info['data_path'].format(episode_chunk=i//new_info['chunks_size'], episode_index=i)
        left, right = pq.read_table(old_path), pq.read_table(new_path)
        assert left.drop(['episode_index','index']).equals(right.drop(['episode_index','index']), check_metadata=False)
        video_hashes = {}
        for key, feature in info['features'].items():
            if feature['dtype'] == 'video':
                old_video = dataset / info['video_path'].format(episode_chunk=old_index//info['chunks_size'], episode_index=old_index, video_key=key)
                new_video = target / new_info['video_path'].format(episode_chunk=i//new_info['chunks_size'], episode_index=i, video_key=key)
                old_sha, new_sha = sha(old_video), sha(new_video)
                assert old_sha == new_sha
                video_hashes[key] = new_sha
        proofs.append(dict(reference_file=row['reference_file'],source_episode=old_index,output_episode=i,
                           non_index_columns_identical=True,video_sha256=video_hashes))
    assert not (target/'meta/stats.json').exists() and not (target/'meta/relative_stats.json').exists()
    result = dict(time=time.time(),episodes=173,frames=new_info['total_frames'],source_dataset=str(dataset),
                  output_dataset=str(target),normalization='Recompute on complete cumulative dataset before training',
                  actual_training_started=False,proofs=proofs)
    (prefix/'built.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='proofs'}), flush=True)


if __name__ == '__main__':
    main()
