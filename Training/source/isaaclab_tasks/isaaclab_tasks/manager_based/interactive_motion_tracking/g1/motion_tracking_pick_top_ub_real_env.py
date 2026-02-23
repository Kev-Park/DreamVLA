# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import right_hand_state_target_reward, joint_deviation_ref_l1, orientation_tracking_error, target_orientation_error, target_ref, root_below_threshold, root_angle_below_threshold, current_time_enc, reset_root_state_for_motion
import numpy as np
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedRLEnvCfg
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base_real import target_z, hand_state_target, hand_state_target_1, rel_pos_goal, keypts_deviation_ref_l2, position_tracking_error, velocity_error, G1InteractiveBaseEnvCfg, object_above_threshold, reset_object_state, object_approach_reward_left, object_approach_reward_right, rel_pose_object, hand_pose, target_orientation, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ActionsCfg as ActionsCfgBase, EventCfg as EventCfgBase, MySceneCfg as MySceneCfgBase
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder, JointNamesOrder_UB
from isaaclab_assets import G1_MINIMAL_CFG_FIXED_BASE as G1_MINIMAL_CFG # isort: skip

JOINTS_MASK = [
    0, # left_hip_pitch_joint
    0, # left_hip_roll_joint
    0, # left_hip_yaw_joint
    0, # left_knee_joint
    0, # left_ankle_pitch_joint
    0, # left_ankle_roll_joint
    0, # right_hip_pitch_joint
    0, # right_hip_roll_joint
    0, # right_hip_yaw_joint
    0, # right_knee_joint
    0, # right_ankle_pitch_joint
    0, # right_ankle_roll_joint
    1, # waist_yaw_joint
    1, # left_shoulder_pitch_joint
    1, # left_shoulder_roll_joint
    1, # left_shoulder_yaw_joint
    1, # left_elbow_joint
    1, # left_wrist_roll_joint
    1, # left_wrist_pitch_joint
    1, # left_wrist_yaw_joint
    1, # right_shoulder_pitch_joint
    1, # right_shoulder_roll_joint
    1, # right_shoulder_yaw_joint
    1, # right_elbow_joint
    1, # right_wrist_roll_joint
    1, # right_wrist_pitch_joint
    1, # right_wrist_yaw_joint
]

KEYPTS_MASK = [
    1, # pelvis
    0, # pelvis_contour_link
    0, # left_hip_pitch_link
    0, # left_hip_roll_link
    0, # left_hip_yaw_link
    0, # left_knee_link
    0, # left_ankle_pitch_link
    0, # left_ankle_roll_link
    0, # right_hip_pitch_link
    0, # right_hip_roll_link
    0, # right_hip_yaw_link
    0, # right_knee_link
    0, # right_ankle_pitch_link
    0, # right_ankle_roll_link
    1, # waist_yaw_link
    1, # waist_roll_link
    1, # torso_link
    1, # logo_link
    1, # head_link
    1, # waist_support_link
    1, # imu_link
    1, # d435_link
    1, # mid360_link
    1, # left_shoulder_pitch_link
    1, # left_shoulder_roll_link
    1, # left_shoulder_yaw_link
    1, # left_elbow_link
    1, # left_wrist_roll_link
    1, # left_wrist_pitch_link
    1, # left_wrist_yaw_link
    1, # left_rubber_hand
    1, # right_shoulder_pitch_link
    1, # right_shoulder_roll_link
    1, # right_shoulder_yaw_link
    1, # right_elbow_link
    1, # right_wrist_roll_link
    1, # right_wrist_pitch_link
    1, # right_wrist_yaw_link
    1, # right_rubber_hand
]

@configclass
class EventCfg(EventCfgBase):
    """Configuration for events."""

    reset_object = EventTerm(
        func=reset_object_state,
        mode="reset",
        params={
            "offset": [0.013, 0.0],
            "height": 1.05,
        }
    )

    reset_base = EventTerm(
        func=reset_root_state_for_motion,
        mode="reset",
        params={
            "offset_z": 0.01
        }
    )

@configclass
class G1Rewards(G1RewardsBase):
    """Reward terms for the MDP."""

    # task terms
    
    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
    
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
    

    
    right_hand_state_target_reward_val = RewTerm(
        func=right_hand_state_target_reward,
        weight=0.3)
    
    object_above_threshold_val = RewTerm(
        func=object_above_threshold,
        weight=.1,
        params={
            "height_thres": 1.05,
            "fall_thres": 1.01,
        }
    )


@configclass
class CommandsCfg_eval:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        resampling_time_range=(10., 10.),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.34, 0.44), # 0.25 - 0.45
            pos_y=(-0.3, 0.0),
            pos_z=(0.292, 0.296),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),  # depends on end-effector axis
            yaw=(-0., 0.),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder_UB, preserve_order=True)})
        
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder_UB, preserve_order=True)})
        
        actions = ObsTerm(func=mdp.last_action)
        curr_time = ObsTerm(func=current_time_enc)       

        rel_pose_object = ObsTerm(func=rel_pose_object, params={"fix_height": True})


        right_hand_state_target_val = ObsTerm(
            func=hand_state_target)

        right_hand_state_target_val_1 = ObsTerm(
            func=hand_state_target_1)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JointNamesOrder_UB, preserve_order=True, scale=0.5, use_default_offset=True)
    
    left_hand_action = mdp.BinaryJointPositionActionCfg(asset_name="robot",
            joint_names=["left_hand.*"],
            open_command_expr={"left_hand.*": 0.0},
            close_command_expr={"left_hand_index_.*": -np.pi / 2.0, 
                                "left_hand_middle_.*": -np.pi / 2.0,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": np.pi / 3.0,
                                "left_hand_thumb_2_joint": np.pi/2.,
                                })

    right_hand_action = mdp.BinaryJointPositionActionCfg(asset_name="robot",
            joint_names=["right_hand.*"],
            open_command_expr={"right_hand.*": 0.0},
            close_command_expr={"right_hand_index_.*": np.pi / 2.0, 
                                "right_hand_middle_.*": np.pi / 2.0,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -np.pi / 3.0,
                                "right_hand_thumb_2_joint": -np.pi/2.,
                                })

@configclass
class MySceneCfg(MySceneCfgBase):
    """Configuration for the terrain scene with a legged robot."""
    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(.05, .15, 0.08),collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.6, 0.2), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.2,
                dynamic_friction=1.2,
                restitution=0.0,
            )
        ),  
    )

    platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.31+1./2., 0, 0.9/2.), rot=(1., 0., 0., 0.)),
        spawn=sim_utils.CuboidCfg(
            size=(1., 2., 0.9),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.6, 0.2), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ))
    )



@configclass
class G1PickTopUBRealEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.ref_motions_path = "../TrajGen/sample/Pick_Top_real1"
        
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005

@configclass
class G1PickTopUBRealEnvEvalCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    commands: CommandsCfg_eval = CommandsCfg_eval()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.ref_motions_path = "../TrajGen/sample/Pick_Top_real1"
        
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005

