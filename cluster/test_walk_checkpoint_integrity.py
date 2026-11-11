import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from walk_checkpoint_integrity import file_info, validate_checkpoint


class CheckpointIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
        (self.root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(8))
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'weight': 'model.safetensors'}, 'metadata': {'total_size': 8}}))
        (self.root / 'trainer_state.json').write_text('{"global_step": 5000}')
        for name in ('config.json', 'processor_config.json', 'statistics.json', 'embodiment_id.json'):
            (self.root / name).write_text('{"fixture": true}')
        for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            with zipfile.ZipFile(self.root / name, 'w') as archive:
                archive.writestr('archive/data.pkl', b'bounded fixture')

    def test_complete_checkpoint_passes_without_loading_arrays(self):
        value = validate_checkpoint(self.root, 5000)
        self.assertEqual(value['tensor_payload_bytes'], 8)
        self.assertEqual(len(value['training_archives']), 3)
        self.assertEqual(len(value['configurations']), 4)

    def test_missing_processor_normalization_cannot_publish_ready(self):
        (self.root / 'statistics.json').unlink()
        with self.assertRaises(FileNotFoundError):
            validate_checkpoint(self.root, 5000)

    def test_incomplete_processor_callback_cannot_publish_ready(self):
        (self.root / 'processor_config.json').write_text('{')
        with self.assertRaises(json.JSONDecodeError):
            validate_checkpoint(self.root, 5000)

    def test_official_processor_subdirectory_fallback_is_supported(self):
        processor = self.root / 'processor'
        processor.mkdir()
        for name in ('processor_config.json', 'statistics.json', 'embodiment_id.json'):
            (self.root / name).rename(processor / name)
        value = validate_checkpoint(self.root, 5000)
        self.assertIn('processor/statistics.json', value['configurations'])

    def test_zero_filled_shard_with_expected_size_fails(self):
        path = self.root / 'model.safetensors'
        path.write_bytes(bytes(path.stat().st_size))
        with self.assertRaisesRegex(AssertionError, 'Invalid tensor header'):
            validate_checkpoint(self.root, 5000)

    def test_expected_size_zero_filled_optimizer_fails(self):
        path = self.root / 'optimizer.pt'
        path.write_bytes(bytes(path.stat().st_size))
        with self.assertRaises(zipfile.BadZipFile):
            validate_checkpoint(self.root, 5000)

    def test_truncated_tensor_payload_fails(self):
        path = self.root / 'model.safetensors'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(AssertionError, 'Incomplete tensor shard'):
            validate_checkpoint(self.root, 5000)

    def test_missing_rng_cannot_pass_as_resumable(self):
        (self.root / 'rng_state.pth').unlink()
        with self.assertRaises(FileNotFoundError):
            validate_checkpoint(self.root, 5000)

    def test_wrong_step_cannot_publish_ready(self):
        with self.assertRaisesRegex(AssertionError, 'Incorrect checkpoint update count'):
            validate_checkpoint(self.root, 10000)

    def test_unwritten_file_with_large_logical_length_requires_review(self):
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=1_000_000, st_blocks=0)):
            with self.assertRaisesRegex(AssertionError, 'Unexpected sparse checkpoint file'):
                file_info(self.root / 'model.safetensors')


if __name__ == '__main__':
    unittest.main()
