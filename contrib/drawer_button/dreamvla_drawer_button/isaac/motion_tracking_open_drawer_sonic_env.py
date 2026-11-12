"""Open Drawer with a SONIC latent body-action space."""

import torch

from isaaclab.assets import Articulation
from isaaclab.sim import PinholeCameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from .robot import G1_SONIC_CFG
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder

from .motion_tracking_open_drawer_env import (
    ActionsCfg as OpenDrawerActionsCfg,
    EventCfg as OpenDrawerEventCfg,
    G1OpenDrawerEnvCfg,
    G1Rewards as OpenDrawerRewardsCfg,
    open_drawer_bonus,
)
from .sonic_action import SONIC_BODY_JOINTS, SonicLatentActionCfg


from ..paths import ASSET_ROOT

_SONIC_DECODER = ASSET_ROOT / "sonic_release/model_decoder.onnx"
_SONIC_ENCODER = ASSET_ROOT / "sonic_release/model_encoder.onnx"
NVIDIA_CABINET_USD = (
    f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
)
# The stock handle's near grasping surface is 37.154 mm shallower than the
# enlarged DreamControl handle when both cabinet roots coincide.  Move the
# stock cabinet toward the robot by that amount so the existing reference
# wrist path meets the same contact plane.  Phase-1 normalizes every motion's
# grasp point to a fixed world X, so this belongs in cabinet placement rather
# than in the trajectory generator.
NVIDIA_CABINET_ROOT_OFFSET_X = 0.425 - 0.037154
NVIDIA_CABINET_ROOT_OFFSET_Z = -0.215 - 0.00075


@configclass
class SonicActionsCfg(OpenDrawerActionsCfg):
    """64-D body latent residual plus one primitive per dexterous hand."""

    joint_pos = SonicLatentActionCfg(
        asset_name="robot",
        decoder_path=str(_SONIC_DECODER),
        encoder_path=str(_SONIC_ENCODER),
        joint_names=SONIC_BODY_JOINTS,
        residual_scale=0.25,
        latent_clip=(-1.0, 1.0),
    )


def reset_sonic_joints_for_motion(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the 43-DoF robot while applying the legacy 27-DoF reference pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if hasattr(env, "start_motion_times"):
        env.motion_ids[env_ids] = torch.randint(
            0, env.total_motions, (len(env_ids),), device=env.device
        )
        motion_times = env.start_motion_times[env_ids].to(
            device=env.device, dtype=torch.float32
        )
        motion_ids = env.motion_ids[env_ids].to(device=env.device, dtype=torch.long)
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        joint_pos[:, asset_cfg.joint_ids] = motion_res["dof_pos"]

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


@configclass
class SonicEventCfg(OpenDrawerEventCfg):
    reset_robot_joints = EventTerm(
        func=reset_sonic_joints_for_motion,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JointNamesOrder, preserve_order=True
            )
        },
    )


def drawer_progress_reward(
    env, asset_cfg: SceneEntityCfg, full_open_distance: float = 0.10
) -> torch.Tensor:
    """Dense privileged reward for physically pulling the target drawer."""
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times
    motion = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    normalized_progress = torch.clamp(drawer_pos / full_open_distance, 0.0, 1.0)
    return normalized_progress * motion["is_closed"].float()


@configclass
class SonicRewardsCfg(OpenDrawerRewardsCfg):
    """Task-scaled rewards so PPO optimizes opening, not only imitation."""

    drawer_progress = RewTerm(
        func=drawer_progress_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])
        },
    )
    open_drawer_bonus = RewTerm(
        func=open_drawer_bonus,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])
        },
    )


@configclass
class G1OpenDrawerSonicEnvCfg(G1OpenDrawerEnvCfg):
    actions: SonicActionsCfg = SonicActionsCfg()
    events: SonicEventCfg = SonicEventCfg()
    rewards: SonicRewardsCfg = SonicRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_SONIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1OpenDrawerSonicNvidiaEnvCfg(G1OpenDrawerSonicEnvCfg):
    """SONIC PPO task whose training physics use NVIDIA's Sektion cabinet."""

    def __post_init__(self):
        super().__post_init__()
        # This must live in the environment configuration rather than being a
        # rollout-only override.  It ensures PPO learns against the exact USD
        # (contacts, rigid-body properties, joints, and visuals) used later by
        # dataset collection and closed-loop GR00T evaluation.
        self.scene.cabinet.spawn.usd_path = NVIDIA_CABINET_USD
        self.events.reset_cabinet.params.update(
            root_offset_x=NVIDIA_CABINET_ROOT_OFFSET_X,
            root_offset_z=NVIDIA_CABINET_ROOT_OFFSET_Z,
        )


@configclass
class G1OpenDrawerSonicNvidiaMultiplicativeEnvCfg(G1OpenDrawerSonicNvidiaEnvCfg):
    """Separate PPO experiment: bounded +/-10% reference-token modulation."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.residual_transform = "multiplicative"
        self.actions.joint_pos.residual_scale = 0.1


@configclass
class G1OpenDrawerSonicNvidiaMultiplicativeFreeEnvCfg(
    G1OpenDrawerSonicNvidiaMultiplicativeEnvCfg
):
    """Separate diagnostic: scale 0.1 without the tanh +/-10% restriction.

    The full decoder token is still clipped to [-1, 1]. This is not compatible
    with additive or tanh-bounded policy checkpoints without retraining.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.residual_transform = "multiplicative_free"


@configclass
class G1OpenDrawerSonicRecordEnvCfg(G1OpenDrawerSonicEnvCfg):
    """SONIC task with a deployable, robot-mounted 640x480 ego camera."""

    def __post_init__(self):
        super().__post_init__()
        # The base task renders every 5 ms physics substep.  Dataset frames are
        # sampled at the 20 ms policy rate, so rendering once per policy step
        # preserves every recorded frame while avoiding three redundant RTX
        # renders per action.
        self.sim.render_interval = self.decimation
        self.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/d435_link/Camera",
            spawn=PinholeCameraCfg(
                focal_length=9.1,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            data_types=["rgb"],
            height=480,
            width=640,
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(0.5, 0.5, -0.5, -0.5),
                convention="opengl",
            ),
        )


@configclass
class G1OpenDrawerSonicMultiplicativeRecordEnvCfg(G1OpenDrawerSonicRecordEnvCfg):
    """Camera-enabled counterpart of the bounded-multiplicative NVIDIA task."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.residual_transform = "multiplicative"
        self.actions.joint_pos.residual_scale = 0.1
        self.scene.cabinet.spawn.usd_path = NVIDIA_CABINET_USD
        self.events.reset_cabinet.params.update(
            root_offset_x=NVIDIA_CABINET_ROOT_OFFSET_X,
            root_offset_z=NVIDIA_CABINET_ROOT_OFFSET_Z,
        )
