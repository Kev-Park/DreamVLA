# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import keypts_deviation_ref_l2, joint_deviation_ref_l1, position_tracking_error, orientation_tracking_error, right_hand_state_target_reward, target_ref, root_below_threshold, root_angle_below_threshold, current_time_enc
import numpy as np
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import G1InteractiveBaseEnvCfg, hand_state_target, hand_state_target_1, rel_pose_object_w_link, object_approach_reward_right, rel_pose_object, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ObservationsCfg as ObservationsCfgBase, ActionsCfg as ActionsCfgBase, EventCfg as EventCfgBase, MySceneCfg as MySceneCfgBase
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import CameraCfg



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


def reset_cabinet(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset the asset root state to the root state of the first pose in the motion library.

    """
    if hasattr(env, 'start_motion_times'):
        motion_times = env.start_motion_times[env_ids]
        motion_ids = env.motion_ids[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        velocities = torch.zeros((env.scene.num_envs, 6), device=env.device)[env_ids]
        
        offset = motion_res["offsets"]
        goal_pos = motion_res["grab_pos"]
        if "cabinet" in env.scene.keys():
            cabinet: RigidObject = env.scene["cabinet"]
            cabinet_pos = goal_pos + offset + env.scene.env_origins[env_ids]
            cabinet_pos[:,0] += 0.425
            cabinet_pos[:,2] -= 0.215
            
            cabinet_quat = torch.zeros((env.scene.num_envs, 4), device=env.device)[env_ids]
            cabinet_quat[:, 3] = 1.0  # set the w component to 1.0 for identity quaternion
            cabinet.write_root_pose_to_sim(torch.cat([cabinet_pos, cabinet_quat], dim=-1), env_ids=env_ids)
            cabinet.write_root_velocity_to_sim(velocities, env_ids=env_ids)
            cabinet_quat = cabinet.data.root_quat_w[env_ids]


@configclass
class EventCfg(EventCfgBase):
    """Configuration for events."""

    reset_cabinet = EventTerm(
        func=reset_cabinet,
        mode="reset"
    )


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer.

    The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
    """
    if env.scene[asset_cfg.name] is None:
        return torch.zeros((env.scene.num_envs,), device=env.device)
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    if env.num_envs < 1001 :
        env.n_successes += (drawer_pos>0.05) * motion_res["is_closed"]
    reward = (drawer_pos>0.05) * motion_res["is_closed"].float()   # clamp the position to [0, 1]
    return reward


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
            weight=-0.1,
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

        right_hand_state_target_reward_val = RewTerm(
            func=right_hand_state_target_reward,
            weight=0.3)


    if TASK_DENSE:
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        object_approach_reward = RewTerm(
            func=object_approach_reward_right,
            weight=100.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
        )

    # Task specific rewards
    if TASK_SPARSE:
        open_drawer_bonus = RewTerm(
            func=open_drawer_bonus,
            weight=.1,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
        )
    


@configclass
class TerminationsCfg(TerminationsCfgBase):
    """Termination terms for the MDP."""
    if TRACKING:
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis","torso_link","waist_yaw_link","waist_roll_link","left_shoulder_pitch_link","right_shoulder_pitch_link",
                                                                            "left_wrist_yaw_link", "right_wrist_yaw_link",
                                                                            "left_hand_index_0_link", "right_hand_index_0_link",
                                                                            "left_hand_index_1_link", "right_hand_index_1_link",
                                                                            "left_hand_middle_0_link", "right_hand_middle_0_link",
                                                                            "left_hand_middle_1_link", "right_hand_middle_1_link",
                                                                            "left_hand_thumb_0_link", "right_hand_thumb_0_link",
                                                                            "left_hand_thumb_1_link", "right_hand_thumb_1_link",
                                                                            "left_hand_thumb_2_link", "right_hand_thumb_2_link",
                                                    ]), "threshold": 1.0},
        )
    torso_below_threshold = DoneTerm(
        func=root_below_threshold, params={"thres": 0.25})
    torso_angle_below_threshold = DoneTerm(
        func=root_angle_below_threshold, params={"thres": 0.3})


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

        if TASK_DENSE:
            curr_time = ObsTerm(func=current_time_enc)       

        rel_pose_object = ObsTerm(func=rel_pose_object)
        rel_pose_object_w_link_val = ObsTerm(func=rel_pose_object_w_link, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        
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
            open_command_expr={"right_hand_index_0_joint": 0, 
                                "right_hand_middle_0_joint": 0,
                                "right_hand_index_1_joint": np.pi/2.,
                                "right_hand_middle_1_joint": np.pi/2.,
                                "right_hand_thumb_.*": 0.0,
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

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/cabinet.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.55, -0.2, 0.83),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=1000.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=.1,
                friction=0.,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    camera = None
    kitchen = None


@configclass
class G1OpenDrawerEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Open_Drawer_sim2"
        

@configclass
class G1MotionTrackingEnvPlayCfg(ManagerBasedRLEnvCfg):
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
        self.ref_motions_path = "../TrajGen/sample/open_drawer_video_"
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
                                          pos=(-1.132+2.21-0.02, -0.843-2.254, 1.863),
                                          rot=(0.7953, 0.58983, -0.0834, -0.11246),
                                          convention="opengl"
                                      ),)
        
        self.scene.kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            spawn=sim_utils.UsdFileCfg(usd_path="assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat3.usd",scale=(1.,1.,1.)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.21-0.09, -2.44, 0.), rot=(1, 0, 0, 0)),
        )

        self.scene.cabinet = None
        self.scene.terrain = None
        
        
