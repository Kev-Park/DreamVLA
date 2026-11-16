"""Standing-only button press; distinct task/checkpoints from walking button PPO.

Uses the native hit_pos_real references (75 ms/frame, no 2.1 m target shift).
The floating-base G1 and SONIC controller remain physical: feet/root are not
locked. Original timed fingertip success is retained; stance is also audited.
"""

import gymnasium as gym
import torch
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul
from .motion_tracking_button_press_sonic_env import (
    G1ButtonPressSonicEnvCfg,
    G1ButtonPressSonicPPORunnerCfg,
)
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import (
    reset_root_state_for_motion,
)

TASK = "DreamVLA-DrawerButton-Button-Press-Standing-SONIC-v0"


def reset_standing_root(
    env,
    env_ids,
    offset_z=0.02,
    xy_range=0.01,
    yaw_range=0.034906585,
    standard_env_count=0,
):
    reset_root_state_for_motion(env, env_ids, offset_z=offset_z)
    if not hasattr(env, "start_motion_times"):
        return
    robot = env.scene["robot"]
    pose = robot.data.root_state_w[env_ids, :7].clone()
    randomize = (env_ids >= standard_env_count).float()
    pose[:, :2] += (
        torch.empty((len(env_ids), 2), device=env.device).uniform_(-xy_range, xy_range)
        * randomize[:, None]
    )
    yaw = (
        torch.empty(len(env_ids), device=env.device).uniform_(-yaw_range, yaw_range)
        * randomize
    )
    rotation = torch.zeros_like(pose[:, 3:7])
    rotation[:, 0] = torch.cos(yaw / 2)
    rotation[:, 3] = torch.sin(yaw / 2)
    pose[:, 3:7] = quat_mul(rotation, pose[:, 3:7])
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)


def stationary_feet_error(env, asset_cfg):
    robot = env.scene["robot"]
    ref = env.motion_lib.get_motion_state(
        env.motion_ids, torch.zeros_like(env.start_motion_times)
    )
    # Native 27-DoF FK order: left/right ankle_roll at indices 7 and 13.
    target = ref["global_keypts"][:, [7, 13], :2] + env.scene.env_origins[:, None, :2]
    actual = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    return torch.linalg.norm(actual - target, dim=-1).mean(dim=-1)


@configclass
class G1ButtonPressStandingSonicEnvCfg(G1ButtonPressSonicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.func = reset_standing_root
        self.events.reset_base.params = dict(
            offset_z=0.02, xy_range=0.01, yaw_range=0.034906585
        )
        self.rewards.stationary_feet = RewardTermCfg(
            func=stationary_feet_error,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                )
            },
        )


@configclass
class G1ButtonPressStandingPPORunnerCfg(G1ButtonPressSonicPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sonic_button_press_standing"


def register_task():
    if TASK not in gym.registry:
        gym.register(
            TASK,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:G1ButtonPressStandingSonicEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}:G1ButtonPressStandingPPORunnerCfg",
            },
        )
