"""Actual new-HDF5 conversion, serial/sharded merge and token roundtrip gate.

Run only after approval for the separate CPU conversion environment. This does
not install packages, shorten recordings, alter codecs or reuse cached outputs.
GR00T normalization generation and official-loader checks remain separate gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, action='append', required=True)
    parser.add_argument('--converter-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Run CPU-only in the approved cluster conversion environment')
    sources = [p.expanduser().resolve(strict=True) for p in args.hdf5]
    if len(sources) != 2 or len(set(sources)) != 2:
        parser.error('Select exactly two distinct complete recordings')
    converter = args.converter_root.expanduser().resolve(strict=True) / 'gear_sonic/scripts/convert_isaac_hdf5_to_lerobot.py'
    assert converter.is_file()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    receipt = dict(started=time.time(), completed=False, commands=[],
                   converter_sha256=digest(converter),
                   source_hashes={str(p): digest(p) for p in sources},
                   normalization_validated=False, official_loader_validated=False,
                   scope='Full actual recordings; serial conversion versus individually converted and merged episodes')

    def run(label, command):
        with (output/(label+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        receipt['commands'].append(dict(label=label, command=command, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError('Inspect failed phase: '+label)

    def convert(label, paths):
        inputs = output/(label+'_inputs')
        inputs.mkdir()
        for i, path in enumerate(paths):
            (inputs/f'{i:06d}__{path.name}').symlink_to(path)
        destination = output/label/'dataset'
        run(label, [sys.executable, str(converter), '--hdf5-root', str(inputs),
                    '--output-path', str(destination), '--dataset-name', 'dataset',
                    '--task-prompt', 'Walk to the object and pick it up.', '--fps', '50'])
        return destination

    try:
        import h5py
        import numpy as np
        import pyarrow.parquet as pq
        import torch
        assert torch.version.cuda is None, 'Use the approved CPU-only profile'
        torch.set_num_threads(1)
        full = convert('full', sources)
        shards = [convert(f'shard{i}', [source]) for i, source in enumerate(sources)]
        merged = output/'merged/dataset'
        run('merge', [sys.executable, str(root/'cluster/merge_lerobot.py'),
                      '--shards', *map(str, shards), '--out', str(merged)])
        run('verify', [sys.executable, str(root/'cluster/verify_merge.py'),
                       '--merged', str(merged), '--reference', str(full)])
        frames = []
        for episode, source in enumerate(sources):
            with h5py.File(source, 'r') as hdf:
                recorded = hdf['data/demo_0/obs/motion_token'][()].astype(np.float32)
            table = pq.read_table(merged/f'data/chunk-000/episode_{episode:06d}.parquet')
            actual = np.asarray(table['action.motion_token'].to_pylist(), dtype=np.float32)
            np.testing.assert_array_equal(actual, recorded)
            assert actual.ndim == 2 and actual.shape[1] == 64 and np.isfinite(actual).all()
            frames.append(len(actual))
        info = json.loads((merged/'meta/info.json').read_text())
        assert info['total_episodes'] == 2 and info['total_frames'] == sum(frames)
        for dataset in (full, merged):
            assert not any((dataset/'meta'/name).exists() for name in ('stats.json', 'relative_stats.json'))
        assert all(digest(p) == receipt['source_hashes'][str(p)] for p in sources)
        receipt.update(completed=True, episode_frames=frames, source_unchanged=True,
                       exact_recorded_tokens=True, cached_normalization_absent=True)
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['finished'] = time.time()
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
