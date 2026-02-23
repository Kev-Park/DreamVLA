import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

import yaml
import pyrealsense2 as rs
from grid_cortex_client.cortex_client import CortexClient
import cv2
from deploy_real_base import *

LOC_X = -0.04
LOC_Y = 0.04
LOC_Z = 0.34

CLOSED_CONFIG = {
    "LeftThumbBend" : 0.,
    "LeftIndex" : 0.,
    "LeftMiddle" : 0.,
    "LeftRing" : 0.,
    "LeftPinky" : 0.,
    "LeftThumbRotation" : 0.,        
    "RightThumbBend" : 0.,
    "RightIndex" : 0.,
    "RightMiddle" : 0.,
    "RightRing" : 0.,
    "RightPinky" : 0.,
    "RightThumbRotation" : 0.        
}

OPEN_CONFIG = {
    "LeftThumbBend" : 1.,
    "LeftIndex" : 1.,
    "LeftMiddle" : 1.,
    "LeftRing" : 1.,
    "LeftPinky" : 1.,
    "LeftThumbRotation" : 0.,        
    "RightThumbBend" : 1.,
    "RightIndex" : 1.,
    "RightMiddle" : 1.,
    "RightRing" : 1.,
    "RightPinky" : 1.,
    "RightThumbRotation" : 0.        
}


POLICY_TIME = 10.0
# TARGET_LOC = [0., 0., 0.8257, 1., 0., 0., 0.]
# TARGET_LOC = [0., 0., 0.5860, 1., 0., 0., 0.]
TARGET_LOC = [0., 0., 0.6550, 1., 0., 0., 0.]

SQUAT_TOUCH_TIME = 5.2

obj = "none"


class ControllerSquat(Controller):
    def run(self, exec_cmd=True):
        self.counter += 1
        t1 = time.time()
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        for i in range(len(self.active_ids)):
            self.qj[i] = self.low_state.motor_state[self.active_ids[i]].q
            self.dqj[i] = self.low_state.motor_state[self.active_ids[i]].dq

        if self.config["imu_type"] == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[0].q
            waist_yaw_omega = self.low_state.motor_state[0].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.default_angles[self.active_ids]) * self.config["dof_pos_scale"]
        dqj_obs = dqj_obs * self.config["dof_vel_scale"]
        return_val = False
        if (time.time() - self.curr_time) > POLICY_TIME and not self.added:
            return_val = True
        

        obs = np.zeros(self.config["num_obs"], dtype=np.float32)
        num_actions = self.config["num_actions"]
        count = time.time() - self.curr_time
        
        period = 10.0
        phase = min(count, period) / period
        sin_phase = np.sin(2 * np.pi * phase)
        right_hand_state = (count > SQUAT_TOUCH_TIME) * 1.0
        right_hand_state_1 = (count < SQUAT_TOUCH_TIME + 0.7) * (count > SQUAT_TOUCH_TIME - 0.7) * 1.0

        obs[:3] = ang_vel
        obs[3:3+3] = gravity_orientation
        obs[3+3:3+3+len(self.active_ids)] = qj_obs
        obs[3+3+len(self.active_ids):3+3+2*len(self.active_ids)] = dqj_obs
        obs[3+3+2*len(self.active_ids) : 3+3+2*len(self.active_ids)+num_actions] = self.action
        obs[3+3+2*len(self.active_ids)+num_actions : 3+3+2*len(self.active_ids)+num_actions+2] = np.array([phase, sin_phase])
        obs[3+3+2*len(self.active_ids)+num_actions+2 : 3+3+2*len(self.active_ids)+num_actions+2+3] = np.array(self.target_loc)[:3]
        obs[3+3+2*len(self.active_ids)+num_actions+2+3 : 3+3+2*len(self.active_ids)+num_actions+2+3+1] = np.array([right_hand_state])
        obs[3+3+2*len(self.active_ids)+num_actions+2+3+1 : 3+3+2*len(self.active_ids)+num_actions+2+3+1+1] = np.array([right_hand_state_1])

        
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = self.action[:-2] * self.config["action_scale"] + self.default_angles[self.active_ids]
        self.hands.publish_cmd()
        for i in range(len(self.active_ids)):
            motor_idx = self.active_ids[i]
            self.low_cmd.motor_cmd[motor_idx].q = SMOOTHNESS_FACTOR*(target_dof_pos[i] - self.qj[i]) + self.qj[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        
        # send the command
        if exec_cmd:
            self.send_cmd(self.low_cmd)
        t2 = time.time()
        time_taken = t2 - t1
        if time_taken < self.config["control_dt"]-0.0015:
            time.sleep(self.config["control_dt"]-0.0015-time_taken)
        
        return return_val



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)
    hands = UnitreeInspire()

    hands.grasp("left")
    hands.grasp("right")
    controller = Controller(config, hands)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()
    # Move to the default position
    controller.move_to_default_pos()
    
    target_loc = None
    print("Target location from vision:", target_loc)

    controller.default_pos_state()

    controller.curr_time = time.time()
    for i in range(5):
        controller_done = controller.run_stand_still(exec_cmd=False)
    controller.action *= 0.
    controller.curr_time = time.time()
    while True:
        try:
            controller_done = controller.run_stand_still()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                create_damping_cmd(controller.low_cmd)
                controller.send_cmd(controller.low_cmd)
                print("Exit")
                exit(0)
            if controller_done or controller.remote_controller.button[KeyMap.B] == 1:
                break
        except KeyboardInterrupt:
            create_damping_cmd(controller.low_cmd)
            controller.send_cmd(controller.low_cmd)
            print("Exit")
            exit(0)
            break
        
    controller.curr_time = time.time()
    while True:
        try:
            controller_done = controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                create_damping_cmd(controller.low_cmd)
                controller.send_cmd(controller.low_cmd)
                print("Exit")
                exit(0)
            # Press the select key to exit
            if controller_done :
                break
        except KeyboardInterrupt:
            create_damping_cmd(controller.low_cmd)
            controller.send_cmd(controller.low_cmd)
            print("Exit")
            exit(0)
            break
    
    controller.curr_time = time.time()
    while True:
        try:
            controller_done = controller.run_stand_still()
            if controller.remote_controller.button[KeyMap.select] == 1:
                create_damping_cmd(controller.low_cmd)
                controller.send_cmd(controller.low_cmd)
                print("Exit")
                exit(0)
            # Press the select key to exit
            if controller_done :
                break
        except KeyboardInterrupt:
            create_damping_cmd(controller.low_cmd)
            controller.send_cmd(controller.low_cmd)
            print("Exit")
            exit(0)
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
