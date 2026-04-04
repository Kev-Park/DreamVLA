# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import keypts_deviation_ref_l2, joint_deviation_ref_l1, position_tracking_error, orientation_tracking_error, target_orientation_error, left_hand_state_target_reward, target_ref, root_below_threshold, root_angle_below_threshold, current_time_enc
import numpy as np
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import G1InteractiveBaseEnvCfg, hand_state_target, hand_state_target_1, rel_pose_object_w_link, object_above_threshold, reset_object_state, object_approach_reward_left, hand_pose, target_orientation, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ActionsCfg as ActionsCfgBase, EventCfg as EventCfgBase, MySceneCfg as MySceneCfgBase
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder


VISUALIZE_MARKERS = True
TRACKING = True
TASK_SPARSE = True
TASK_DENSE = False

JOINTS_MASK = [
    1, # left_hip_pitch_joint
    1, # left_hip_roll_joint
    1, # left_hip_yaw_joint
    1, # left_knee_joint
    0, # left_ankle_pitch_joint
    0, # left_ankle_roll_joint
    1, # right_hip_pitch_joint
    1, # right_hip_roll_joint
    1, # right_hip_yaw_joint
    1, # right_knee_joint
    0, # right_ankle_pitch_joint
    0, # right_ankle_roll_joint
    1, # waist_yaw_joint
    1, # left_shoulder_pitch_joint
    1, # left_shoulder_roll_joint
    1, # left_shoulder_yaw_joint
    1, # left_elbow_joint
    0, # left_wrist_roll_joint
    0, # left_wrist_pitch_joint
    0, # left_wrist_yaw_joint
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
    1, # pelvis_contour_link
    1, # left_hip_pitch_link
    1, # left_hip_roll_link
    1, # left_hip_yaw_link
    1, # left_knee_link
    0, # left_ankle_pitch_link
    1, # left_ankle_roll_link
    1, # right_hip_pitch_link
    1, # right_hip_roll_link
    1, # right_hip_yaw_link
    1, # right_knee_link
    0, # right_ankle_pitch_link
    1, # right_ankle_roll_link
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
    0, # left_rubber_hand
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
    add_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.0, 0.9),
            "operation": "add",
        },
    )

    reset_object = EventTerm(
        func=reset_object_state,
        mode="reset",
        params={
            "height": 0.1,
            "offset": [0.12, -0.1]
        }
    )




@configclass
class G1Rewards(G1RewardsBase):
    """Reward terms for the MDP."""

    if TRACKING:
        joint_deviation_ref = RewTerm(
            func=joint_deviation_ref_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "joint_mask": JOINTS_MASK})
        
        keypts_deviation_ref = RewTerm(
            func=keypts_deviation_ref_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "keypts_mask": KEYPTS_MASK})
        

        # pose_deviation
        position_tracking_error = RewTerm(
            func=position_tracking_error,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        orientation_tracking_error = RewTerm(
            func=orientation_tracking_error,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        target_orientation_error = RewTerm(func=target_orientation_error,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"])},
            weight=-1.0)
    
    if TASK_DENSE:
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        object_approach_reward = RewTerm(func=object_approach_reward_left,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]), "offset_x": 0.12, "offset_y": -0.1},
            weight=100.0)
        
    
    left_hand_state_target_reward_val = RewTerm(
        func=left_hand_state_target_reward,
        weight=0.3)
    
    if TASK_SPARSE:
        object_lifted_reward = RewTerm(
            func=object_above_threshold,
            weight=1.,
            params={"height_thres": 0.2, "fall_thres": 0.15}
        )

    

    
@configclass
class TerminationsCfg(TerminationsCfgBase):
    """Termination terms for the MDP."""
    torso_below_threshold = DoneTerm(
        func=root_below_threshold, params={"thres": 0.15})
    torso_angle_below_threshold = DoneTerm(
        func=root_angle_below_threshold, params={"thres": 0.4})
    

@configclass
class ObservationsCfg():
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
        
        actions = ObsTerm(func=mdp.last_action)
        if TRACKING:        
            target_ref_curr = ObsTerm(func=target_ref, params={"visualize_markers": VISUALIZE_MARKERS})
            target_ref_next = ObsTerm(func=target_ref, params={"time_offset": 0.2})
            target_ref_next_next = ObsTerm(func=target_ref, params={"time_offset": 0.4})
        
        if TASK_DENSE:
            curr_time = ObsTerm(func=current_time_enc)
        
        rel_pose_object_w_link_val = ObsTerm(func=rel_pose_object_w_link, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"])})
        hand_pose_val = ObsTerm(func=hand_pose, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"])})
        target_orientation = ObsTerm(
            func=target_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"])})

        left_hand_state_target_val = ObsTerm(
            func=hand_state_target)
        left_hand_state_target_val_1 = ObsTerm(
            func=hand_state_target_1)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg(ActionsCfgBase):
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
            open_command_expr={"right_hand_index_.*": np.pi / 2.0, 
                                "right_hand_middle_.*": np.pi / 2.0,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -np.pi / 3.0,
                                "right_hand_thumb_2_joint": -np.pi/2.,
                                },
            close_command_expr={"right_hand_index_.*": np.pi / 2.0, 
                                "right_hand_middle_.*": np.pi / 2.0,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -np.pi / 3.0,
                                "right_hand_thumb_2_joint": -np.pi/2.,
                                })

@configclass
class MySceneCfg(MySceneCfgBase):
    """Configuration for the terrain scene with a legged robot."""

    kitchen = None
    
    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(.05, .05, 0.4),collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.6, 0.2), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            )
        ),
    )



@configclass
class G1GroundPickEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Ground_Pick_sim2"
        
                

@configclass
class G1GroundPickEnvPlayCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain = None
        self.ref_motions_path = "../TrajGen/sample/Ground_Pick_sim2"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("Robot added")
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005

        self.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera_new",
                                      spawn=PinholeCameraCfg(
                                          focal_length=18.1476,
                                          focus_distance=400.,
                                          horizontal_aperture=20.955,
                                          clipping_range=(0.1, 10000.0),
                                      ),
                                      data_types=["rgb"],
                                      height=1920,
                                      width=2560,
                                      offset=CameraCfg.OffsetCfg(
                                          pos=(-2.33+3.1-0.034, -4.45+0.9, 1.81),
                                          rot=(0.7538, 0.61221, -0.1505, -0.1853),
                                          convention="opengl"
                                      ),)
        # self.scene.kitchen.init_state.pos = (2.55, 0, 0.38)
        self.ref_motions_path = "../TrajGen/sample/Ground_Pick_sim2"
        self.scene.terrain = None
        self.scene.sky_light = None
        
        self.scene.kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            spawn=sim_utils.UsdFileCfg(usd_path="assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat.usd",scale=(1.,1.,0.85)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(3.1-0.06, 0.9, 0.), rot=(1, 0, 0, 0)),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/pitcher_big.usd",
                scale=(1., 1., 1.2),
                mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            ),
        )
        

