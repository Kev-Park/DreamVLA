"""Exercise the approved mask repair through the actual reward function."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'Training/source/isaaclab_tasks/isaaclab_tasks'
SOURCE = ROOT/'manager_based/motion_tracking/g1/motion_tracking_env.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'joint_deviation_ref_l1')
mask = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'JOINTS_MASK' for t in n.targets))
motion = ast.parse((ROOT/'utils/motion_lib/motion_lib_base.py').read_text(encoding='utf-8'))
names = next(ast.literal_eval(n.value) for n in motion.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'JointNamesOrder' for t in n.targets))
scope = dict(torch=torch, ManagerBasedRLEnv=object, Articulation=object, List=list,
             SceneEntityCfg=lambda name: NS(name=name, joint_ids=slice(None)),
             JOINTS_MASK=mask, JointNamesOrder=names)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), scope)


class JointRewardMask(unittest.TestCase):
    def evaluate(self, joints, reference, weights):
        env = NS(scene={'robot': NS(data=NS(joint_pos=joints))},
                 episode_length_buf=torch.zeros(len(joints)), step_dt=.02,
                 start_motion_times=torch.zeros(len(joints)), device='cpu', motion_ids=None,
                 motion_lib=NS(get_motion_state=lambda *a: {'dof_pos': reference}))
        return scope['joint_deviation_ref_l1'](env, joint_mask=weights)

    def test_old_weights_keep_old_joint_contributions(self):
        generator = torch.Generator().manual_seed(52)
        joints = torch.randn(128, 29, generator=generator, dtype=torch.float64)
        reference = torch.randn(128, 29, generator=generator, dtype=torch.float64)
        keep = [i for i, name in enumerate(names) if name not in ('waist_roll_joint', 'waist_pitch_joint')]
        weights = torch.arange(1, 28, dtype=torch.float64)/27
        joints[:, names.index('waist_roll_joint')] = 1000
        expected = ((joints-reference)[:, keep]*weights).abs().sum(1)
        actual = self.evaluate(joints, reference, weights.tolist())
        # The production function constructs weights in the default Torch dtype.
        expected = ((joints-reference)[:, keep]*weights.float()).abs().sum(1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_existing_29_weights_and_27_layout_are_unchanged(self):
        for count in (27, 29):
            joints = torch.arange(count, dtype=torch.float64).reshape(1, count)
            weights = list(range(1, count+1))
            expected = (joints*torch.tensor(weights)).sum(1)
            torch.testing.assert_close(self.evaluate(joints, joints*0, weights), expected, atol=0, rtol=0)

    def test_zero_and_malformed_masks(self):
        joints = torch.ones(2, 29)
        torch.testing.assert_close(self.evaluate(joints, joints*0, [0]*27), torch.zeros(2))
        with self.assertRaisesRegex(ValueError, 'layout differ'):
            self.evaluate(joints, joints*0, [1]*26)


if __name__ == '__main__':
    unittest.main()
