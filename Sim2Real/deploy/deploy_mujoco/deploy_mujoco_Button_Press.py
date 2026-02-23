import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from deploy_mujoco_base import *



if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR) + "Button_Press.pt"
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        all_joints = config["joint_names"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        left_hand_open_ja = np.array(config.get("left_hand_closed_joint_angles"))
        right_hand_open_ja = np.array(config.get("right_hand_closed_joint_angles"))
        left_hand_closed_ja = np.array(config.get("left_hand_closed_joint_angles"))
        right_hand_closed_ja = np.array(config.get("right_hand_closed_joint_angles"))
        right_hand_closed_ja[-2:] = 0.0  # Set the last two joints to 0 for the right hand
        right_hand_open_ja[-2:] = 0.0  # Set the last two joints to 0 for the right hand
        
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        left_hand_joints = config.get("left_hand_joint_names")
        right_hand_joints = config.get("right_hand_joint_names")

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path+"scene_full_button_press.xml")
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    print(m.opt.solver)
    # load policy
    policy = torch.jit.load(policy_path)
    active_joints = ["left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint", 
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint", 
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint"]
    
    active_joint_ids = [all_joints.index(name) for name in active_joints]
    left_hand_joint_ids = [all_joints.index(name) for name in left_hand_joints]
    right_hand_joint_ids = [all_joints.index(name) for name in right_hand_joints]
    target_dof_pos = default_angles.copy()[active_joint_ids]
    target_dof_left_hand = default_angles.copy()[left_hand_joint_ids]
    target_dof_right_hand = default_angles.copy()[right_hand_joint_ids]
    button_press_time = 2.8  # seconds
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        rel_pose_object = [0.45, -0.2, 0.2, 1., 0., 0., 0.] # Get this
        d.qpos = default_angles
        object_pose = np.array(rel_pose_object) + np.array([0.03, 0., 0.753, 0., 0., 0., 0.])
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        m.body_pos[body_id] = object_pose[:3]
        m.body_quat[body_id] = object_pose[3:7]
        mujoco.mj_step(m, d)
        counter = 0.
        while viewer.is_running() :
            # Initialize the robot from the default position.
            
            step_start = time.time()
            if counter*simulation_dt > 7. :
                rel_pose_object[0] = np.random.uniform(0.43, 0.54) # Get this
                rel_pose_object[1] = np.random.uniform(-0.27, -0.04)
                rel_pose_object[2] = np.random.uniform(0.13, 0.29)
                object_pose = np.array(rel_pose_object) + np.array([0.03, 0., 0.753, 0., 0., 0., 0.])
                body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
                m.body_pos[body_id] = object_pose[:3]
                m.body_quat[body_id] = object_pose[3:7]
                d.qpos[7:] = default_angles
                d.qvel[6:] = 0.
                d.qpos[:3] = np.array([0., 0., 0.833])
                d.qpos[3:7] = np.array([1., 0., 0., 0.])
                d.qvel[:6] = 0.
            tau = pd_control(target_dof_pos, d.qpos[7:][active_joint_ids], kps[active_joint_ids], np.zeros_like(kds[active_joint_ids]), d.qvel[6:-6][active_joint_ids], kds[active_joint_ids])
            tau_left_hand = pd_control(target_dof_left_hand, d.qpos[7:][left_hand_joint_ids], kps[left_hand_joint_ids], np.zeros_like(kds[left_hand_joint_ids]), d.qvel[6:-6][left_hand_joint_ids], kds[left_hand_joint_ids])
            tau_right_hand = pd_control(target_dof_right_hand, d.qpos[7:][right_hand_joint_ids], kps[right_hand_joint_ids], np.zeros_like(kds[right_hand_joint_ids]), d.qvel[6:-6][right_hand_joint_ids], kds[right_hand_joint_ids])
            # print(d.ctrl)
            d.ctrl[:] = 0.
            d.ctrl[active_joint_ids] = tau
            d.ctrl[left_hand_joint_ids] = tau_left_hand
            d.ctrl[right_hand_joint_ids] = tau_right_hand
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            # print(tau)
            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
                # create observation
                qj = d.qpos[7:][active_joint_ids]
                dqj = d.qvel[6:-6][active_joint_ids]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                qj = (qj - default_angles[active_joint_ids]) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                
                count = counter * simulation_dt
                period = 10.0
                phase = min(count, period) / period
                sin_phase = np.sin(2 * np.pi * phase)
                right_hand_state = (count > button_press_time) * 1.0
                right_hand_state_1 = (count < button_press_time + 0.7) * (count > button_press_time - 0.7) * 1.0
                obs[:3] = omega
                obs[3:3+3] = gravity_orientation
                obs[3+3:3+3+len(active_joint_ids)] = qj
                obs[3+3+len(active_joint_ids):3+3+2*len(active_joint_ids)] = dqj
                obs[3+3+2*len(active_joint_ids) : 3+3+2*len(active_joint_ids)+num_actions] = action
                obs[3+3+2*len(active_joint_ids)+num_actions : 3+3+2*len(active_joint_ids)+num_actions+2] = np.array([phase, sin_phase])
                obs[3+3+2*len(active_joint_ids)+num_actions+2 : 3+3+2*len(active_joint_ids)+num_actions+2+3] = np.array(rel_pose_object)[:3]
                obs[3+3+2*len(active_joint_ids)+num_actions+2+3 : 3+3+2*len(active_joint_ids)+num_actions+2+3+1] = np.array([right_hand_state])
                obs[3+3+2*len(active_joint_ids)+num_actions+2+3+1 : 3+3+2*len(active_joint_ids)+num_actions+2+3+1+1] = np.array([right_hand_state_1])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action[:-2] * action_scale + default_angles[active_joint_ids]
                target_dof_left_hand = left_hand_open_ja * (1 - action[-2]) + left_hand_closed_ja * action[-2]
                target_dof_right_hand = right_hand_open_ja * (1 - action[-1]) + right_hand_closed_ja * action[-1]

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
    
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
