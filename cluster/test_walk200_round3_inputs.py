"""Reject partial or mismatched students before collection consumes GPU time."""
import json
from pathlib import Path
import tempfile
import unittest

from adroit.walk200_round3_inputs import EPISODES, FRAMES, PREFIX, STEPS, TAG, student_inputs


class StudentHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.info = self.root / PREFIX / 'meta/info.json'
        self.receipt = self.root / 'dline' / TAG / (TAG + '_r2.trained.json')
        self.checkpoint = self.root / 'checkpoints' / (TAG + '_r2') / f'checkpoint-{STEPS}'
        self.state = self.checkpoint / 'trainer_state.json'
        self.write(self.info, {'total_frames': FRAMES, 'total_episodes': EPISODES})
        self.write(self.receipt, {'run': TAG + '_r2', 'steps': STEPS, 'frames': FRAMES,
                                 'episodes': EPISODES, 'batch_size': 16,
                                 'checkpoint': str(self.checkpoint), 'dataset': str(self.root / PREFIX)})
        self.write(self.state, {'global_step': STEPS})
        self.write(self.checkpoint / 'model.safetensors.index.json',
                   {'weight_map': {'parameter': 'model-00001-of-00001.safetensors'}})
        (self.checkpoint / 'model-00001-of-00001.safetensors').write_bytes(b'fixture-not-model-weights')

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def change(self, path, **values):
        obj = json.loads(path.read_text())
        obj.update(values)
        self.write(path, obj)

    def test_completed_metadata(self):
        self.assertEqual(student_inputs(self.root)[0], self.checkpoint)
        self.assertEqual(STEPS, 92967)

    def test_recovery_checkpoint_is_rejected(self):
        self.change(self.state, global_step=5000)
        with self.assertRaises(AssertionError):
            student_inputs(self.root)

    def test_matching_fabricated_smaller_budget_is_rejected(self):
        self.change(self.info, total_frames=14035)
        self.change(self.receipt, frames=14035, steps=5000)
        self.change(self.state, global_step=5000)
        with self.assertRaises(AssertionError):
            student_inputs(self.root)

    def test_receipt_must_exist_and_match_genealogy(self):
        self.change(self.receipt, dataset='/other/dataset')
        with self.assertRaises(AssertionError):
            student_inputs(self.root)
        self.receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            student_inputs(self.root)

    def test_missing_or_escaping_shard_is_rejected(self):
        (self.checkpoint / 'model-00001-of-00001.safetensors').unlink()
        with self.assertRaises(FileNotFoundError):
            student_inputs(self.root)
        self.write(self.checkpoint / 'model.safetensors.index.json',
                   {'weight_map': {'parameter': '../model.safetensors'}})
        with self.assertRaises(AssertionError):
            student_inputs(self.root)


if __name__ == '__main__':
    unittest.main()
