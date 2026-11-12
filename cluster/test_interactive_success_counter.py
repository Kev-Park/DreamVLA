"""Optional evaluation counters must not change rewards or prevent stepping."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1'


def load(filename, name):
    source = ROOT/filename
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(torch=torch, ManagerBasedRLEnv=object, SceneEntityCfg=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope[name]


class OptionalSuccessCounter(unittest.TestCase):
    def test_button_reset_with_optional_contact_history(self):
        reset = load('motion_tracking_button_press_env.py', 'reset_button')
        for history in (False, True):
            poses, velocities = [], []
            class Scene(dict):
                num_envs = 3
                env_origins = torch.tensor([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
            scene = Scene(buttons=NS(write_root_pose_to_sim=lambda p, **kw: poses.append(p),
                                     write_root_velocity_to_sim=lambda p, **kw: velocities.append(p)))
            env = NS(start_motion_times=torch.zeros(3), motion_ids=torch.zeros(3),
                     device='cpu', scene=scene,
                     motion_lib=NS(get_motion_state=lambda *a: {'offsets': torch.ones(2, 3),
                                                                'grab_pos': torch.zeros(2, 3)}))
            if history:
                env.last_contacts = torch.ones(3, 2)
            reset(env, torch.tensor([0, 2]))
            torch.testing.assert_close(poses[0], torch.tensor([[2.03, 1., .932, 1., 0., 0., 0.],
                                                              [4.03, 1., .932, 1., 0., 0., 0.]]))
            torch.testing.assert_close(velocities[0], torch.zeros(2, 6))
            if history:
                torch.testing.assert_close(env.last_contacts, torch.tensor([[0., 0.], [1., 1.], [0., 0.]]))

    def test_counter_presence_preserves_reward_and_existing_tally(self):
        positions = torch.tensor([0., .049, .05, .051, .2])
        closed = torch.tensor([True, True, True, True, False])
        for filename, name in [('motion_tracking_open_drawer_env.py', 'open_drawer_bonus'),
                               ('motion_tracking_open_door_env.py', 'open_door_bonus')]:
            function = load(filename, name)
            for count_enabled in (False, True):
                env = NS(scene={'cabinet': NS(data=NS(joint_pos=positions[:, None]))},
                         episode_length_buf=torch.zeros(5), start_motion_times=torch.zeros(5),
                         step_dt=.02, motion_ids=None, num_envs=5,
                         motion_lib=NS(get_motion_state=lambda *a: {'is_closed': closed}))
                if count_enabled:
                    env.n_successes = torch.ones(5)
                actual = function(env, NS(name='cabinet', joint_ids=[0]))
                expected = torch.tensor([0., 0., 0., 1., 0.])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if count_enabled:
                    torch.testing.assert_close(env.n_successes, expected+1, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
