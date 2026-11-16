"""Exercise actual launcher device resolution without starting Isaac or CUDA."""
import ast
import contextlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]/'Training/source/isaaclab/isaaclab/app/app_launcher.py'
CLASS = next(n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body
             if isinstance(n, ast.ClassDef) and n.name == 'AppLauncher')
METHOD = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == '_resolve_device_settings')


def resolve(args, modern):
    scope = {'os': os, 'sys': sys,
             'SimulationApp': SimpleNamespace(DEFAULT_LAUNCHER_CONFIG={'active_cuda_gpus': None} if modern else {}),
             'AppLauncher': SimpleNamespace(_APPLAUNCHER_CFG_INFO={'device': ([str], 'cuda:0')})}
    exec(compile(ast.Module(body=[METHOD], type_ignores=[]), str(SOURCE), 'exec'), scope)
    owner = SimpleNamespace(_xr=False)
    with contextlib.redirect_stdout(io.StringIO()):
        scope['_resolve_device_settings'](owner, args)
    return args


class DeviceRouting(unittest.TestCase):
    def test_masked_single_gpu_keeps_cuda_ordinal_not_physical_ordinal(self):
        # On this mask logical cuda:1 is physical GPU 6. Passing active_gpu=1
        # would instead choose physical graphics GPU 1.
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5,6'}):
            args = resolve({'device': 'cuda:1', 'multi_gpu': False}, modern=True)
        self.assertEqual(args['physics_gpu'], 1)
        self.assertEqual(args['active_cuda_gpus'], [1])
        self.assertIsNone(args['active_gpu'])
        self.assertEqual(args['max_gpu_count'], 1)

    def test_legacy_sdk_selection_is_unchanged(self):
        args = resolve({'device': 'cuda:1', 'multi_gpu': False}, modern=False)
        self.assertEqual(args, {'device': 'cuda:1', 'multi_gpu': False, 'physics_gpu': 1, 'active_gpu': 1})

    def test_intentional_multi_gpu_selection_is_unchanged(self):
        for explicit in ({}, {'multi_gpu': True}):
            args = resolve({'device': 'cuda:1', **explicit}, modern=True)
            self.assertEqual(args['active_gpu'], 1)
            self.assertNotIn('active_cuda_gpus', args)
            self.assertNotIn('max_gpu_count', args)

    def test_distributed_single_device_routes_rank_in_cuda_space(self):
        values = {'LOCAL_RANK': '1', 'JAX_LOCAL_RANK': '0', 'RANK': '3', 'JAX_RANK': '0', 'WORLD_SIZE': '2'}
        with patch.dict(os.environ, values), patch('os.cpu_count', return_value=64), patch('sys.argv', ['probe']):
            args = resolve({'device': 'cuda:0', 'distributed': True}, modern=True)
        self.assertEqual(args['physics_gpu'], 1)
        self.assertEqual(args['active_cuda_gpus'], [1])
        self.assertIs(args['multi_gpu'], False)


if __name__ == '__main__':
    unittest.main()
