# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import keypts_deviation_ref_l2, joint_deviation_ref_l1, position_tracking_error, orientation_tracking_error, target_orientation_error, right_hand_state_target_reward, target_ref, root_below_threshold, root_angle_below_threshold, current_time_enc
import numpy as np
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import G1InteractiveBaseEnvCfg, hand_state_target, hand_state_target_1, reset_object_state, object_approach_reward_right, rel_pose_object, hand_pose, target_orientation, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ObservationsCfg as ObservationsCfgBase, ActionsCfg as ActionsCfgBase, EventCfg as EventCfgBase, MySceneCfg as MySceneCfgBase
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
    1, # left_wrist_roll_joint
    1, # left_wrist_pitch_joint
    1, # left_wrist_yaw_joint
    1, # right_shoulder_pitch_joint
    1, # right_shoulder_roll_joint
    1, # right_shoulder_yaw_joint
    1, # right_elbow_joint
    0, # right_wrist_roll_joint
    0, # right_wrist_pitch_joint
    0, # right_wrist_yaw_joint
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
    1, # left_rubber_hand
    1, # right_shoulder_pitch_link
    1, # right_shoulder_roll_link
    1, # right_shoulder_yaw_link
    1, # right_elbow_link
    1, # right_wrist_roll_link
    1, # right_wrist_pitch_link
    0, # right_wrist_yaw_link
    0, # right_rubber_hand
]
        
@configclass
class EventCfg(EventCfgBase):
    """Configuration for events."""
    add_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.0, 0.4),
            "operation": "add",
        },
    )

    reset_object = EventTerm(
        func=reset_object_state,
        mode="reset",
        params={"height": 0.95, "offset": [-0.04, -0.02]}
    )


def object_lifted_reward(env: ManagerBasedRLEnv, height_thres = 0.74, fall_thres = 0.25) -> torch.Tensor:
    object: RigidObject = env.scene["object"]
    root_pos = object.data.root_pos_w
    has_grasped = (root_pos[:,2] > height_thres).float()

    has_not_fallen = (root_pos[:,2] > fall_thres).float()
    reward = (has_grasped) + (1-has_grasped)*has_not_fallen*(root_pos[:,1]>0.16)
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    if env.num_envs < 1001 :
        env.n_successes += (reward > 0.) * motion_res["is_closed"]
    return reward*motion_res["is_closed"].float() #- 0.3*(1.-motion_res["is_closed"].float())*has_fallen



def goal_approach_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object"), std: float = 1.0, offset_x: float = 0.1, offset_y: float = 0.05) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_object = asset.data.root_pos_w - env.scene.env_origins
    
    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_times_back = 10. + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times_back)
    root_pos = motion_res["global_keypts"][:,-1,:]
    root_pos[:, 0] += offset_x
    root_pos[:, 1] += offset_y
    time_mask = (motion_res["is_closed"].float())
    rel_vector = root_pos_object[:, :2] - root_pos[:, :2]
    dist = torch.norm(rel_vector, dim=1)
    reward = (env.prev_dist1-dist)*time_mask
    env.prev_dist1 = dist
    return reward

@configclass
class G1Rewards(G1RewardsBase):
    """Reward terms for the MDP."""
    if TRACKING : 
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

    if TASK_DENSE:
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        object_approach_reward = RewTerm(func=object_approach_reward_right,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]), "offset_x": 0.1, "offset_y": 0.05},
            weight=100.0)
        goal_approach_reward = RewTerm(func=goal_approach_reward,
            params={"asset_cfg": SceneEntityCfg("object")},
            weight=100.0)
        
    # Task specific rewards
    target_orientation_error = RewTerm(func=target_orientation_error,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
        weight=-1.0)
    
    right_hand_state_target_reward_val = RewTerm(
        func=right_hand_state_target_reward,
        weight=0.3)
    
    if TASK_SPARSE:
        object_lifted_reward = RewTerm(
            func=object_lifted_reward,
            weight=1.0,   
        )

    
@configclass
class TerminationsCfg(TerminationsCfgBase):
    """Termination terms for the MDP."""
    if TRACKING :
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis","torso_link","waist_yaw_link","waist_roll_link","left_shoulder_pitch_link","right_shoulder_pitch_link",
                                                    ]), "threshold": 1.0},
        )
    torso_below_threshold = DoneTerm(
        func=root_below_threshold, params={"thres": 0.3})
    torso_angle_below_threshold = DoneTerm(
        func=root_angle_below_threshold, params={"thres": 0.5})


def rel_pose_target(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    if not hasattr(env, 'motion_lib'):
        # If motion library is not available, return a tensor of zeros
        obs = torch.zeros((env.scene.num_envs, 7), device=env.device)
        obs[:, 4] = 1.0  # set the w component to 1.0 for identity quaternion
        return obs
    else :
    
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
        motion_times_back = 10. + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times_back)
        root_pos = motion_res["global_keypts"][:,-1,:]
        
        root_pos_robot_local = math_utils.quat_apply(
            math_utils.quat_conjugate(root_rot_robot), root_pos - root_pos_robot)
        root_rot_robot_local = math_utils.quat_mul(
            math_utils.quat_conjugate(root_rot_robot), root_rot)
        
        # Create the target reference tensor
        target_ref_tensor = torch.cat([root_pos_robot_local, root_rot_robot_local], dim=-1)
        
        return target_ref_tensor



@configclass
class ObservationsCfg(ObservationsCfgBase):
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

        # Task specific observations :-
        right_hand_pose_val = ObsTerm(func=hand_pose, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        if TASK_DENSE:
            rel_pose_object = ObsTerm(func=rel_pose_object)
            rel_pose_target = ObsTerm(func=rel_pose_target)
            curr_time = ObsTerm(func=current_time_enc)

        target_orientation = ObsTerm(
            func=target_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})

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
class ActionsCfg(ActionsCfgBase):
    """Action specifications for the MDP."""
    left_hand_action = mdp.BinaryJointPositionActionCfg(asset_name="robot",
            joint_names=["left_hand.*"],
            open_command_expr={"left_hand_index_.*": -np.pi / 2.0, 
                                "left_hand_middle_.*": -np.pi / 2.0,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": np.pi / 3.0,
                                "left_hand_thumb_2_joint": np.pi/2.,
                                },
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

    platform1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[2.18, 1.82, 0.27], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(.5, .5, .54),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.6, 0.2), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ))
    )

    platform2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[2.25, -0.09, 0.29], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(.5, .5, .58),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.6, 0.2), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ))
    )
    
    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(.05, .05, 0.25),collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0., 0.5), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            )
        ),
    )
    office = None
    

@configclass
class G1PickPlaceEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Pick_Place_sim2"
        

@configclass
class G1PickPlacePlayEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        self.ref_motions_path = "../TrajGen/sample/Pick_Place_sim2"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("Robot added")
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        rot_before = torch.tensor([[0.26009, 0.17922, 0.53836, 0.78128]])
        rot_mat_before = math_utils.matrix_from_quat(rot_before)
        rot_mat_after = rot_mat_before.clone()
        rot_mat_after[:,0,:] = -rot_mat_before[:,0,:]  # flip the x-axis
        rot_mat_after[:,1,:] = -rot_mat_before[:,1,:]
        rot_after = math_utils.quat_from_matrix(rot_mat_after)

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
                                          pos=(-3.9042+2.1+1.5, -46.92+44.5, 1.678),
                                          rot=rot_after[0],
                                          convention="opengl"
                                      ),)
        
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/pitcher.usd",
                scale=(.7, .7, 1.),
                mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            ),
        )
        
        self.scene.platform1 = None
        self.scene.platform2 = None

        self.scene.office = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Office",
            spawn=sim_utils.UsdFileCfg(usd_path="assets/office1.usd",scale=(1.,1.,.98)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.1+1.55, 44.5, 0.0001), rot=(0, 0, 0, 1)),
        )
        
        
