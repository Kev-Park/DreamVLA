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
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import CameraCfg
from isaac_utils.rotations import(
    slerp,
)
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import G1InteractiveBaseEnvCfg, hand_state_target, hand_state_target_1, rel_pose_object_w_link, object_above_threshold, reset_object_state, rel_pose_object, hand_pose, object_approach_reward_right, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ActionsCfg as ActionsCfgBase, MySceneCfg as MySceneCfgBase, EventCfg as EventCfgBase
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
        params={
            "height": 1.0,
            "offset": [0.0, 0.0],
        },
        mode="reset"
    )



def target_orientation_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) - 1.
    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    root_pos = motion_res["grab_pos"] + motion_res["offsets"]
    time_mask = 1. - motion_res["is_closed"].float()
    
    time_init = torch.clip((motion_times+1.)/1.5, 0., 1.)  # Ensure the time mask is between 0 and 1
    
    x_axis = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()
    z_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()

    target_rot_mat_init = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
    target_rot_quat_init = math_utils.quat_from_matrix(target_rot_mat_init)

    a_axis = root_pos - root_pos_link # (2*x_axis + y_axis)/sqrt(5)
    a_axis = a_axis / torch.norm(a_axis, dim=1, keepdim=True)
    
    b_axis = torch.zeros_like(a_axis)
    b_axis[:,0] = -a_axis[:,1]
    b_axis[:,1] = a_axis[:,0]
    b_axis[:,2] = 0.0
    b_axis = b_axis / torch.norm(b_axis, dim=1, keepdim=True)

    # z_axis is a_axis x b_axis
    z_axis = torch.cross(a_axis, b_axis, dim=1)

    x_axis = 2*a_axis - b_axis
    x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

    y_axis = torch.cross(z_axis, x_axis, dim=1)

    target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
    target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
    
    target_rot_quat = slerp(target_rot_quat_init, target_rot_quat, time_init.unsqueeze(1))
    angle = math_utils.quat_error_magnitude(target_rot_quat, root_rot_link)

    z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis_post)
    angle_post = torch.acos(torch.clamp(z_axis_w[:, 2], -1.0, 1.0))
    return torch.abs(angle) * time_mask + torch.abs(angle_post) * (1. - time_mask)


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

        right_hand_state_target_reward_val = RewTerm(
            func=right_hand_state_target_reward,
            weight=0.3)
        
        target_orientation_error = RewTerm(func=target_orientation_error,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
            weight=-1.0)


    if TASK_DENSE:
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        object_approach_reward = RewTerm(func=object_approach_reward_right,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
            weight=100.0)
        right_hand_state_target_reward_val = RewTerm(
            func=right_hand_state_target_reward,
            weight=0.3)
    
    
    if TASK_SPARSE:
        object_above_the_ground = RewTerm(
            func=object_above_threshold,
            weight=.5,    
            params={"height_thres": 0.95, "fall_thres": 0.9}
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
    
    
def target_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    if not hasattr(env, 'motion_lib'):
        return_val = torch.zeros((env.scene.num_envs, 4), device=env.device)
        return_val[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        return return_val
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) - 1.
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        time_mask = 1. - motion_res["is_closed"].float()
        time_init = torch.clip((motion_times+1.)/1.5, 0., 1.)  # Ensure the time mask is between 0 and 1
        
        x_axis = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()
        z_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()

        target_rot_mat_init = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat_init = math_utils.quat_from_matrix(target_rot_mat_init)

        a_axis = root_pos - root_pos_link # (2*x_axis + y_axis)/sqrt(5)
        a_axis = a_axis / torch.norm(a_axis, dim=1, keepdim=True)
        
        b_axis = torch.zeros_like(a_axis)
        b_axis[:,0] = -a_axis[:,1]
        b_axis[:,1] = a_axis[:,0]
        b_axis[:,2] = 0.0
        b_axis = b_axis / torch.norm(b_axis, dim=1, keepdim=True)

        z_axis = torch.cross(a_axis, b_axis, dim=1)

        x_axis = 2*a_axis - b_axis
        x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

        y_axis = torch.cross(z_axis, x_axis, dim=1)

        target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
        
        target_rot_quat = slerp(target_rot_quat_init, target_rot_quat, time_init.unsqueeze(1))

        x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis_post = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        
        target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
        target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)        
        return target_rot_quat * time_mask.unsqueeze(1) + target_post_rot_quat * (1. - time_mask.unsqueeze(1))


@configclass
class ObservationsCfg:
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
            target_ref_next = ObsTerm(func=target_ref, params={"time_offset": .1})
            target_ref_next_next = ObsTerm(func=target_ref, params={"time_offset": .2})
            target_orientation_hand = ObsTerm(
                func=target_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        
        if TASK_DENSE:
            current_time = ObsTerm(func=current_time_enc)
        
        right_hand_state_target_val = ObsTerm(
            func=hand_state_target)
        right_hand_state_target_val_1 = ObsTerm(
            func=hand_state_target_1)


        # Task specific observations:
        rel_pose_object = ObsTerm(func=rel_pose_object)
        rel_pose_object_w_link_val = ObsTerm(func=rel_pose_object_w_link, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        right_hand_pose_val = ObsTerm(func=hand_pose, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        object_mass = ObsTerm(func=mdp.rigid_body_mass, params={"asset_cfg": SceneEntityCfg("object")})
        
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

    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=sim_utils.CuboidCfg(
            size=(1., 2., 0.8),collision_props=sim_utils.CollisionPropertiesCfg(),
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
            size=(.05, .05, 0.2),collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.2, 0.6), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.2,
                dynamic_friction=1.2,
                restitution=0.0,
            )
        ),  
    )


@configclass
class G1PickEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.kitchen.init_state.pos = (2.55, 0, 0.4)
        self.ref_motions_path = "../TrajGen/sample/Pick_sim2"



@configclass
class G1PickPlayEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=100.)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
        rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(rot)))
        theta = -np.pi*0.75
        rot_z_theta = np.array([[np.cos(theta), -np.sin(theta), 0.0], \
                                [np.sin(theta), np.cos(theta), 0.0], \
                                [0.0, 0.0, 1.0]])
        rot_mat = rot_z_theta @ rot_mat
        rot_quat = math_utils.quat_from_matrix(torch.tensor(rot_mat))
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
                                          pos=(-1.03+2.1-0.034, 4.05-0.9, 1.31),
                                          rot=rot_quat,
                                          convention="opengl"
                                      ),)
        self.ref_motions_path = "../TrajGen/sample/Pick_sim2"
        self.scene.terrain = None
        self.scene.sky_light = None
        
        self.scene.kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            spawn=sim_utils.UsdFileCfg(usd_path="assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat3.usd",scale=(1.,1.,0.89)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.1-0.06, 1.0, 0.), rot=(1, 0, 0, 0)),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/mustard_bottle.usd",
                scale=(1., 1., 1.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            ),
        )
        # self.scene.terrain = None
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("Robot added")
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
