"""Real parquet/statistics merger checks in the baseline converter environment.

Videos are deliberately opaque byte fixtures: these tests prove copying, not
codec validity or rendering. No simulator, model or real dataset is used.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import datasets
import jsonlines
import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.utils import serialize_dict

from merge_lerobot import merge


DATA = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
VIDEO_KEY = 'observation.images.ego_view'
LENGTHS = [3, 2, 4, 1]


def write_jsonl(path, rows):
    with jsonlines.open(path, 'w') as stream:
        stream.write_all(rows)


def build(root, identities, caches=False):
    meta = root/'meta'
    meta.mkdir(parents=True)
    numeric = {
        'episode_index': {'dtype': 'int64', 'shape': [1]},
        'index': {'dtype': 'int64', 'shape': [1]},
        'frame_index': {'dtype': 'int64', 'shape': [1]},
        'task_index': {'dtype': 'int64', 'shape': [1]},
        'timestamp': {'dtype': 'float32', 'shape': [1]},
        'state.joint_pos': {'dtype': 'float32', 'shape': [2]},
        'action.motion_token': {'dtype': 'float32', 'shape': [64]},
        'action.right_hand_joints': {'dtype': 'float32', 'shape': [7]},
    }
    hf_features = datasets.Features({
        key: (datasets.Value(value['dtype']) if value['shape'] == [1]
              else datasets.Sequence(datasets.Value(value['dtype']), length=value['shape'][0]))
        for key, value in numeric.items()})
    episodes, statistics, offset = [], [], 0
    for index, identity in enumerate(identities):
        count = LENGTHS[identity]
        arrays = {
            'episode_index': np.full(count, index), 'index': np.arange(offset, offset+count),
            'frame_index': np.arange(count), 'task_index': np.zeros(count, dtype=np.int64),
            'timestamp': np.arange(count, dtype=np.float32)/50,
            'state.joint_pos': np.full((count, 2), identity+.25, dtype=np.float32),
            'action.motion_token': np.arange(count*64, dtype=np.float32).reshape(count, 64)/16+identity,
            'action.right_hand_joints': np.full((count, 7), identity*.2, dtype=np.float32),
        }
        parquet = root/DATA.format(episode_chunk=index//2, episode_index=index)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_dict(arrays, features=hf_features).to_parquet(str(parquet))
        video = root/VIDEO.format(episode_chunk=index//2, episode_index=index, video_key=VIDEO_KEY)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b'opaque video copy fixture '+str(identity).encode())
        episodes.append({'episode_index': index, 'tasks': ['pick'], 'length': count})
        stat_features = {key: {**value, 'shape': tuple(value['shape'])} for key, value in numeric.items()}
        stats = serialize_dict(compute_episode_stats(arrays, stat_features))
        statistics.append({'episode_index': index, 'stats': stats})
        offset += count
    info = {'features': {**numeric, VIDEO_KEY: {'dtype': 'video', 'shape': [4, 4, 3]}},
            'fps': 50, 'codebase_version': 'v2.1', 'robot_type': 'g1', 'chunks_size': 2,
            'script_config': {'fixture': True}, 'data_path': DATA, 'video_path': VIDEO,
            'total_episodes': len(identities), 'total_frames': offset,
            'total_videos': len(identities), 'total_chunks': (len(identities)+1)//2,
            'splits': {'train': f'0:{len(identities)}'}, 'discarded_episode_indices': []}
    (meta/'info.json').write_text(json.dumps(info))
    (meta/'modality.json').write_text(json.dumps({'state': {'joint_pos': {'start': 0, 'end': 2}}}))
    write_jsonl(meta/'tasks.jsonl', [{'task_index': 0, 'task': 'pick'}])
    write_jsonl(meta/'episodes.jsonl', episodes)
    write_jsonl(meta/'episodes_stats.jsonl', statistics)
    if caches:
        for name in ('stats.json', 'relative_stats.json'):
            (meta/name).write_text('{"wrong_shard_local_quantiles":true}')


class MergerRoundTrip(unittest.TestCase):
    def assert_dataset_equal(self, actual, expected):
        for name in ('info.json', 'modality.json'):
            self.assertEqual(json.loads((actual/'meta'/name).read_text()),
                             json.loads((expected/'meta'/name).read_text()))
        for name in ('tasks.jsonl', 'episodes.jsonl', 'episodes_stats.jsonl'):
            with jsonlines.open(actual/'meta'/name) as left, jsonlines.open(expected/'meta'/name) as right:
                self.assertEqual(list(left), list(right))
        info = json.loads((actual/'meta/info.json').read_text())
        for index in range(info['total_episodes']):
            relative = DATA.format(episode_chunk=index//2, episode_index=index)
            self.assertTrue(pq.read_table(actual/relative).equals(pq.read_table(expected/relative),
                                                                 check_metadata=True), relative)
            relative = VIDEO.format(episode_chunk=index//2, episode_index=index, video_key=VIDEO_KEY)
            self.assertEqual((actual/relative).read_bytes(), (expected/relative).read_bytes())
        for name in ('stats.json', 'relative_stats.json'):
            self.assertFalse((actual/'meta'/name).exists(), 'Shard normalization caches must not propagate')

    def test_contiguous_shards_match_single_dataset_numerically_and_in_video_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, expected, result = (root/name for name in ('first', 'second', 'expected', 'result'))
            build(first, [0, 1], caches=True)
            build(second, [2, 3], caches=True)
            build(expected, [0, 1, 2, 3])
            with contextlib.redirect_stdout(io.StringIO()):
                merge([first, second], result)
            self.assert_dataset_equal(result, expected)
            self.assertTrue((first/'meta/stats.json').is_file(), 'Source datasets must remain untouched')

    def test_episode_subset_keeps_requested_demonstrations_and_recomputes_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, expected, result = (root/name for name in ('source', 'expected', 'result'))
            build(source, [0, 1, 2, 3], caches=True)
            build(expected, [3, 0])
            with contextlib.redirect_stdout(io.StringIO()):
                merge([source], result, [3, 0])
            self.assert_dataset_equal(result, expected)

    def test_mismatched_modality_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, result = (root/name for name in ('first', 'second', 'result'))
            build(first, [0])
            build(second, [1])
            (second/'meta/modality.json').write_text('{"state":{"joint_pos":{"start":1,"end":3}}}')
            with self.assertRaisesRegex(ValueError, 'modality'):
                merge([first, second], result)
            self.assertFalse(result.exists())


if __name__ == '__main__':
    unittest.main()
