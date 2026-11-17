"""CPU regression checks for durable completion and safe shard recovery."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def module(name):
    path = Path(__file__).parent/(name+'.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CompletionTests(unittest.TestCase):
    def test_receipt_precedes_app_shutdown(self):
        wrapper = module('run_expert_entrypoint')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            receipt = base/'receipt.json'
            script = base/'evaluator.py'
            script.write_text(
                'import json\nfrom pathlib import Path\n'
                'class App:\n'
                ' def close(self, wait_for_replicator=False, exit_code=None):\n'
                f'  assert json.loads(Path({str(receipt)!r}).read_text())["completed"]\n'
                '_APP=App()\ndef main(): return 0\n')
            with patch('sys.argv', ['wrapper', '--entrypoint', str(script), '--receipt', str(receipt)]), \
                    patch.object(wrapper.importlib.metadata, 'version', return_value='test'):
                wrapper.main()
            self.assertTrue(json.loads(receipt.read_text())['completed'])

    def test_recovery_requires_full_coverage_and_finite_traces(self):
        runner = module('evaluate_gr00t_students')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base/'eval0.log').write_text('VLA + SONIC EVAL SUMMARY\nCompleted episodes: 2\n')
            keys = ['obj_pos', 'obj_quat', 'root_pos', 'root_quat', 'wrist',
                    'body_pos', 'ref_body_pos', 'ref_root_pos']
            values = {key: np.zeros((2, 3)) for key in keys}
            np.savez(base/'b0_ep0_m0_traj.npz', **values)
            with self.assertRaises(AssertionError):
                runner.completed_shard_files(base, 0, [0, 4])
            np.savez(base/'b0_ep1_m4_traj.npz', **values)
            self.assertEqual(len(runner.completed_shard_files(base, 0, [0, 4])), 3)
            values['obj_pos'][0, 0] = np.nan
            np.savez(base/'b0_ep1_m4_traj.npz', **values)
            with self.assertRaises(AssertionError):
                runner.completed_shard_files(base, 0, [0, 4])


if __name__ == '__main__':
    unittest.main()
