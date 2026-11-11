"""Exercise the real collector scorers on numeric HDF5 fixtures, without Isaac."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]/'Training/scripts/reinforcement_learning/rsl_rl'
sys.path.insert(0, str(SCRIPTS))
tree = ast.parse((SCRIPTS/'collect_sonic_adapter.py').read_text(encoding='utf8'))
names = {'_score_episode_arrays', 'score_written_episode', 'score_rollout_episode'}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(functions) == len(names)
scope = {'np': np}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'collector_scorers', 'exec'), scope)


class CollectorScoring(unittest.TestCase):
    def fixture(self, change=None):
        n = 64
        roots = np.tile(np.array([0., 0., 1.], np.float32), (n, 1))
        quats = np.tile(np.array([1., 0., 0., 0.], np.float32), (n, 1))
        wrist = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        obj = roots + np.array([.1, .1, .1], np.float32)
        oq = quats.copy()
        if change == 'fell': obj[0, 2] = .1
        if change == 'horizontal': oq[10] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
        if change == 'notbox': obj[-1, 0] = 1.
        if change == 'slip': obj[:, 2] += np.linspace(0, .01, n, dtype=np.float32)
        raw = {'object': {'root_pos_w': obj, 'root_quat_w': oq},
               'robot': {'root_pos_w': roots, 'root_quat_w': quats}}
        return raw, {'right_wrist': wrist}

    def compare(self, change):
        raw, teleop = self.fixture(change)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            box = folder/'box.json'
            box.write_text(json.dumps({'lo': [.05, .05, .05], 'hi': [.15, .15, .15]}))
            path = folder/'episode.hdf5'
            with h5py.File(path, 'w') as h:
                group = h.create_group('data/demo_0')
                values = {'obs/object_pos': raw['object']['root_pos_w'],
                          'obs/object_quat': raw['object']['root_quat_w'],
                          'obs/robot0_root_pos_w': raw['robot']['root_pos_w'],
                          'obs/robot0_root_quat_w': raw['robot']['root_quat_w'],
                          'teleop/right_wrist': teleop['right_wrist']}
                for name, value in values.items():
                    group.create_dataset(name, data=value.astype(np.float64))
            calibration = 'file:'+str(box)
            memory = scope['score_rollout_episode'](raw, teleop, calibration)
            recorded = scope['score_written_episode'](path, calibration)
            self.assertEqual(memory, recorded)
            return memory

    def test_float32_in_memory_and_float64_recording_agree_on_success(self):
        result, why = self.compare(None)
        self.assertTrue(result)
        self.assertEqual(why['why'], '-')

    def test_failure_rules_match_on_both_collection_paths(self):
        for change, reason in [('fell', 'fell'), ('horizontal', 'horiz'),
                               ('notbox', 'notbox'), ('slip', 'slip')]:
            with self.subTest(change=change):
                result, why = self.compare(change)
                self.assertFalse(result)
                self.assertIn(reason, why['why'])


if __name__ == '__main__':
    unittest.main()
