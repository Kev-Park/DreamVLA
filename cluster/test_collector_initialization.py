"""Exercise the actual collector setup without allocating models or Isaac."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = (Path(__file__).resolve().parents[1] /
          'Training/scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py')
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                               for t in node.targets)


start = next(i for i, n in enumerate(main.body) if assigns(n, 'grasp_gate'))
stop = next(i for i, n in enumerate(main.body) if assigns(n, 'output_root'))
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_read_motion_ids')


class SuppliedImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return ast.copy_location(ast.Pass(), node)


module = ast.Module(body=[copy.deepcopy(helper), *copy.deepcopy(main.body[start:stop])],
                    type_ignores=[])
module = ast.fix_missing_locations(SuppliedImports().visit(module))
code = compile(module, str(SOURCE), 'exec')


class CollectorInitialization(unittest.TestCase):
    def run_setup(self, ids='1 3', enabled=True):
        models, drivers = [], []

        def model(**kwargs):
            instance = SimpleNamespace(parameters=kwargs)
            models.append(instance)
            return instance

        def driver(env, policy, adapter, **kwargs):
            instance = SimpleNamespace(policy=policy, parameters=kwargs)
            drivers.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'motion_ids.txt'
            path.write_text(ids)
            args = SimpleNamespace(grasp_gate=False, dagger_vla_checkpoint='checkpoint' if enabled else None,
                                   gr00t_service=None,
                                   dagger_demo_root='demos', dagger_motion_ids=str(path),
                                   dagger_rescue=True, dagger_rescue_box='af60', dagger_beta=.25,
                                   dagger_chunk=8, dagger_rollouts=1, dagger_diagnostics=True, dagger_trigger='none',
                                   dagger_trigger_thr=1., dagger_trigger_smooth=3, dagger_trigger_budget=.1,
                                   dagger_trigger_lr=.01, dagger_expert_base='expert', dagger_rescue_blend=0,
                                   dagger_finger_close_thres=.5, seed=17)
            scope = {'args_cli': args, 'env': SimpleNamespace(unwrapped=SimpleNamespace(total_motions=4)),
                     'device': 'cpu', 'load_gr00t_policy': model, 'DaggerDriver': driver,
                     'ObsToPolicyAdapter': lambda *a, **k: object(),
                     'ObsAdapterConfig': lambda **k: k,
                     'SimpleG1RobotModel': SimpleNamespace(build=lambda: object()),
                     'demo_motion_ids': lambda root: [0, 1, 2, 3],
                     'dagger_closed_pose': lambda root: np.ones(7), 'np': np, 'Path': Path, 'os': os}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, scope)
        return models, drivers, scope

    def test_one_model_and_driver_with_validated_reference_order(self):
        models, drivers, scope = self.run_setup('3 1')
        self.assertEqual(len(models), 1, 'A second GR00T load wastes GPU memory and replaces the first driver')
        self.assertEqual(len(drivers), 1)
        self.assertIs(drivers[0].policy, models[0])
        self.assertEqual(scope['dagger_motions'], [3, 1])
        self.assertEqual(drivers[0].parameters['beta'], 0)
        self.assertEqual(drivers[0].parameters['chunk'], 8)
        self.assertTrue(drivers[0].parameters['diagnostics'])

    def test_reference_ids_remain_bounded_by_loaded_library(self):
        with self.assertRaisesRegex(ValueError, 'inside the loaded reference library'):
            self.run_setup('1 4')

    def test_expert_only_collection_does_not_load_gr00t(self):
        models, drivers, scope = self.run_setup(enabled=False)
        self.assertEqual(models, [])
        self.assertEqual(drivers, [])
        self.assertIsNone(scope['dagger'])


if __name__ == '__main__':
    unittest.main()
