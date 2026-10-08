"""Reuse an exact converted prefix; convert only appended HDF5 episodes.

Outputs one merged dataset. Whole-dataset GR00T normalization is regenerated;
cached prefix statistics are never propagated. Existing inputs stay read-only.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def checked_prefix(root, prefix_root, prefix_dataset):
    files = sorted(root.glob('**/*.hdf5'))
    prior = sorted(prefix_root.glob('**/*.hdf5'))
    info = json.loads((prefix_dataset/'meta/info.json').read_text())
    if not prior or len(prior) != info['total_episodes']:
        raise ValueError('Prefix HDF5 count does not match converted episode count')
    if len(files) <= len(prior):
        raise ValueError('Incremental conversion requires new episodes after the prefix')
    if info.get('discarded_episode_indices'):
        raise ValueError('Cannot reuse a prefix with discarded episodes')
    for i, (old, new) in enumerate(zip(prior, files)):
        if old.resolve(strict=True) != new.resolve(strict=True):
            raise ValueError(f'Prefix episode {i} differs: {old} versus {new}')
    return files, len(prior), info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdf5-root', type=Path, required=True)
    p.add_argument('--prefix-hdf5-root', type=Path, required=True)
    p.add_argument('--prefix-dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task-prompt', required=True)
    p.add_argument('--fps', type=int, default=50)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    started = time.monotonic()
    files, reused, info = checked_prefix(a.hdf5_root, a.prefix_hdf5_root, a.prefix_dataset)
    assert a.fps == info['fps'], 'FPS differs from prefix'
    assert not a.out.exists(), 'Refusing to overwrite an output dataset'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='incremental_', dir=a.out.parent))
    inputs = work/'hdf5'; inputs.mkdir()
    for i, source in enumerate(files[reused:]):
        (inputs/f'{i:06d}__{source.name}').symlink_to(source.resolve(strict=True))
    here = Path(__file__).resolve().parent
    converted = work/'converted'/'delta'
    print(f'[incremental] reuse {reused}; convert {len(files)-reused} new episodes', flush=True)
    subprocess.run(['bash',str(here/'convert_sharded.sh'),'--hdf5-root',str(inputs),
        '--recursive','--output-path',str(converted.parent),'--dataset-name','delta',
        '--task-prompt',a.task_prompt,'--fps',str(a.fps),'--shards',str(a.shards)],check=True)
    convert_seconds = time.monotonic()-started
    import os
    wbc = Path(os.environ.get('GR00T_WHOLEBODYCONTROL_DIR',str(Path.home()/'kevin/GR00T-WholeBodyControl')))
    subprocess.run([str(wbc/'.venv_data_collection/bin/python'),str(here/'merge_lerobot.py'),
        '--shards',str(a.prefix_dataset),str(converted),'--out',str(a.out)],check=True)
    result = json.loads((a.out/'meta/info.json').read_text())
    assert result['total_episodes'] == len(files)
    for name in ['stats.json','relative_stats.json']:
        assert not (a.out/'meta'/name).exists(), 'Stale normalization cache must not be copied'
    report = dict(prefix_dataset=str(a.prefix_dataset.resolve()),
        reused_episodes=reused, new_episodes=len(files)-reused,
        episodes=result['total_episodes'], frames=result['total_frames'],
        convert_seconds=convert_seconds, total_seconds=time.monotonic()-started,
        normalization='Regenerate on the complete merged dataset at fine-tune time')
    (a.out.parent/'incremental_manifest.json').write_text(json.dumps(report,indent=2))
    # The final merged dataset contains independent copies of every file.
    shutil.rmtree(work)
    print(f"[done] wrote {result['total_frames']} frames across {result['total_episodes']} episode(s) -> {a.out}",flush=True)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
