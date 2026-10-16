"""Read-only comparison of matched expert demonstrations across budget exports."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def episode_path(dataset, info, index, kind, video_key=None):
    return dataset / info[kind].format(episode_chunk=index // info['chunks_size'],
        episode_index=index, video_key=video_key)


def main():
    request = Path(sys.argv[1])
    spec = json.loads(request.read_text())
    root = Path.home() / 'kevin'
    full = root / spec['full_dataset']
    info = json.loads((full / 'meta/info.json').read_text())
    modality = json.loads((full / 'meta/modality.json').read_text())
    tasks = [json.loads(line) for line in (full / 'meta/tasks.jsonl').read_text().splitlines()]
    columns = [name for name, feature in info['features'].items() if 'float' in feature['dtype']]
    columns += ['frame_index', 'task_index']
    cameras = [name for name, feature in info['features'].items() if feature['dtype'] == 'video']
    result = dict(time=time.time(), comparisons=[], numerical_mismatches=[], video_byte_mismatches=[])
    for subset in spec['subsets']:
        dataset = root / subset['dataset']
        small_info = json.loads((dataset / 'meta/info.json').read_text())
        assert small_info['total_episodes'] == len(subset['episode_mapping'])
        assert info['fps'] == small_info['fps'] == 50
        assert json.loads((dataset / 'meta/modality.json').read_text()) == modality
        assert [json.loads(line) for line in (dataset / 'meta/tasks.jsonl').read_text().splitlines()] == tasks
        for small_index, full_index in enumerate(subset['episode_mapping']):
            left_path = episode_path(dataset, small_info, small_index, 'data_path')
            right_path = episode_path(full, info, full_index, 'data_path')
            left = pd.read_parquet(left_path, columns=columns)
            right = pd.read_parquet(right_path, columns=columns)
            assert len(left) == len(right) == 499
            differences = []
            for column in columns:
                a, b = np.stack(left[column].to_numpy()), np.stack(right[column].to_numpy())
                assert a.shape == b.shape
                assert np.isfinite(a).all() and np.isfinite(b).all()
                if not np.array_equal(a, b):
                    differences.append(dict(column=column, max_abs_difference=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))))
            if differences:
                result['numerical_mismatches'].append(dict(subset=subset['name'],episode=small_index,differences=differences))
            videos = []
            for camera in cameras:
                a = episode_path(dataset, small_info, small_index, 'video_path', camera)
                b = episode_path(full, info, full_index, 'video_path', camera)
                ah, bh = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b))
                identical = ah == bh
                videos.append(dict(camera=camera, byte_identical=identical, subset_sha256=ah, full_sha256=bh))
                if not identical:
                    result['video_byte_mismatches'].append(dict(subset=subset['name'],episode=small_index,camera=camera))
            result['comparisons'].append(dict(subset=subset['name'],subset_episode=small_index,
                full_episode=full_index,frames=len(left),numerical_columns_compared=len(columns),
                numerical_columns_identical=not differences,videos=videos))
    result.update(episodes_compared=len(result['comparisons']),
        all_numerical_columns_identical=not result['numerical_mismatches'],
        all_encoded_videos_byte_identical=not result['video_byte_mismatches'],
        limitations=['Episode/index identifiers are deliberately renumbered in subset datasets and are excluded.',
                     'Different encoded video bytes alone do not prove different decoded pixels; decode mismatches before attributing an input change.'])
    output = root / 'eval_videos/pick_scaling_base_input_comparison.json'
    assert not output.exists(), 'Preserve any existing comparison'
    output.write_text(json.dumps(result,indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['comparisons','numerical_mismatches','video_byte_mismatches']}),flush=True)
    if result['numerical_mismatches']:
        raise SystemExit('Matched numerical inputs differ; inspect the report before drawing conclusions')


if __name__ == '__main__':
    main()
