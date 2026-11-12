"""Original DreamControl Button Press objective with the shared SONIC interface.

This is a fingertip-proximity task, not button-depression physics. No drawer
checkpoint, rewards, cabinet, or trained policy is transferred. The original
pointing-hand primitives and reference-time-gated touch_goal reward are retained.
"""

import gymnasium as gym
from isaaclab.utils import configclass
from .robot import G1_SONIC_CFG
from .motion_tracking_button_press_env import G1ButtonPressEnvCfg
from .motion_tracking_open_drawer_sonic_env import (
    _SONIC_DECODER,
    _SONIC_ENCODER,
    reset_sonic_joints_for_motion,
)
from .sonic_action import SONIC_BODY_JOINTS
from .sonic_teacher_repair import TeacherRepairActionCfg
from .sonic_dreamvla_experiment import G1DreamVlaStylePPORunnerCfg

TASK = "DreamVLA-DrawerButton-Button-Press-SONIC-v0"


@configclass
class G1ButtonPressSonicEnvCfg(G1ButtonPressEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_SONIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = TeacherRepairActionCfg(
            asset_name="robot",
            decoder_path=str(_SONIC_DECODER),
            encoder_path=str(_SONIC_ENCODER),
            joint_names=SONIC_BODY_JOINTS,
            residual_transform="multiplicative_free",
            residual_scale=0.5,
            latent_clip=(-1.0, 15.0 / 16.0),
            token_postprocess="fsq32_lattice_round",
        )
        # Retain the original reference-reset configuration/joint order, but
        # initialize all 43 robot joints, including the two extra SONIC waist DoFs.
        self.events.reset_robot_joints.func = reset_sonic_joints_for_motion
        for value in vars(self.observations.policy).values():
            if hasattr(value, "params") and "visualize_markers" in value.params:
                value.params["visualize_markers"] = False
        self.sim.render_interval = self.decimation


@configclass
class G1ButtonPressSonicPPORunnerCfg(G1DreamVlaStylePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sonic_button_press"
        self.resume = False
        self.save_interval = 50


def register_task():
    if TASK not in gym.registry:
        gym.register(
            TASK,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:G1ButtonPressSonicEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}:G1ButtonPressSonicPPORunnerCfg",
            },
        )
