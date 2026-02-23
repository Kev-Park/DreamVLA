import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

import yaml
import pyrealsense2 as rs
from grid_cortex_client.cortex_client import CortexClient
import cv2
from deploy_real_base_UB import *

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
    "RightThumbBend" : .7,
    "RightIndex" : 1.,
    "RightMiddle" : 1.,
    "RightRing" : 1.,
    "RightPinky" : 1.,
    "RightThumbRotation" : 0.        
}


POLICY_TIME = 11.0
TARGET_LOC = [0.43, -0.06, 0.213, 1., 0., 0., 0.]
# TARGET_LOC = [0.34, -0.227, 0.293, 1., 0., 0., 0.]
PICK_TIME = 4.75
# obj = "pink label on teabag"
# obj = "green canada dry bottle"
obj = "none"


def get_target_point(pipeline):
    profile = pipeline.start(config)

    # Get device intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)
    SKIP_FRAMES = 20
    # Wait for a coherent pair of frames
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames) if align else frames
        color_frame = aligned_frames.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())[:,:,::-1]
        rgb = np.ascontiguousarray(rgb)
        SKIP_FRAMES -= 1
        
        # Get the depth frame
        depth_frame = aligned_frames.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data())
        if SKIP_FRAMES > 0 :
            continue
        if rgb is not None :
            # Detect the object

            data = rgb.copy()
            with CortexClient() as client:
                start = time.time()  # Start timing
                output = client.run(
                    model_id=MODEL_ID, image_input=data, prompt=obj, debug=True
                )
                print(
                    f"Time taken for {MODEL_ID}: {(time.time() - start) * 1000:.2f} ms"
                )  # Log the time taken
                print(f"SUCCESS: Model '{MODEL_ID}' ran successfully.")
                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]
            if boxes is not None and len(boxes) > 0:
                print("scores: ", scores)
                i_max = -1
                y_maxx = 0
                for i in range(len(boxes)):
                    if scores[i] < 0.1:
                        continue
                    # Draw box on the image
                    x_min, y_min, x_max, y_max = boxes[i]
                    if y_max > y_maxx:
                        y_maxx = y_max
                        i_max = i 
                x_min, y_min, x_max, y_max = boxes[i_max]
                if i_max != -1 :    
                    mid_x, mid_y = get_box_center(boxes[i_max], [data.shape[1], data.shape[0]])
                    print("mid_x, mid_y: ", mid_x, mid_y)
                    
                    # Get depth at the center of the box
                    if mid_x != -1 and mid_y != -1:
                        depth_value = depth[int(mid_y), int(mid_x)]
                        print(f"Depth at center ({mid_x}, {mid_y}): {depth_value} mm")
                    else:
                        print("Invalid box center coordinates, skipping depth retrieval.")
                    
                    x_min = max(5, x_min)
                    y_min = max(5, y_min)
                    x_max = min(data.shape[1]-5, x_max)
                    y_max = min(data.shape[0]-5, y_max)
                    print(f"Box coordinates: ({x_min}, {y_min}), ({x_max}, {y_max})")
                    cv2.rectangle(rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.circle(rgb, (int(mid_x), int(mid_y)), 5, (0, 0, 255), -1)
                    cv2.putText(rgb, f"{scores[i_max]:.2f}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Get 3d coordinates of the center of the box
                    if mid_x != -1 and mid_y != -1:
                        depth_in_meters = depth_value / 1000.0
                        x_center_ = (mid_x - color_intrinsics.ppx) / color_intrinsics.fx * depth_in_meters
                        y_center_ = (mid_y - color_intrinsics.ppy) / color_intrinsics.fy * depth_in_meters
                        z_center_ = depth_in_meters
                        x_center = LOC_X + z_center_
                        y_center = LOC_Y - x_center_
                        z_center = LOC_Z - y_center_

                        print(f"3D coordinates of the center before: ({x_center}, {y_center}, {z_center})")
                        x_center = min(0.43,max(0.35,x_center))
                        y_center = min(0.05,max(-0.3,y_center))
                        z_center = 0.293
                        print(f"3D coordinates of the center after: ({x_center}, {y_center}, {z_center})")
                        cv2.imshow("Detected Object", rgb)
                        if cv2.waitKey(0) == ord('q'):
                            print("Exiting...")
                            break
                            
                else :
                    print("No valid box detected")
                    # return None
                cv2.imshow("Detected Object", rgb)
                # Close window on key press
                if cv2.waitKey(0) == ord('q'):
                    print("Exiting...")
                    break
                time.sleep(0.5)
                # break

                
            else:
                print("No boxes detected")
                mid_x, mid_y = -1, -1
        else :
            print("invalid image")    
    
    pipeline.stop()
    return x_center, y_center, z_center


class ControllerPickTop(Controller):
    def run(self):
        self.counter += 1
        t1 = time.time()
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
        period = 15.0
        phase = min(count, period) / period
        sin_phase = np.sin(2 * np.pi * phase)
        right_hand_state =  np.exp(-np.abs(count - 0.475))#(count > PICK_TIME) * 1.0
        right_hand_state_1 =  np.exp(-0.5*np.abs(count - 0.475))#(count < PICK_TIME + 0.7) * (count > PICK_TIME - 0.7) * 1.0

        obs[:3] = gravity_orientation
        obs[3:3+len(self.active_ids)] = qj_obs
        obs[3+len(self.active_ids):3+2*len(self.active_ids)] = dqj_obs
        obs[3+2*len(self.active_ids) : 3+2*len(self.active_ids)+num_actions] = self.action
        obs[3+2*len(self.active_ids)+num_actions : 3+2*len(self.active_ids)+num_actions+2] = np.array([phase, sin_phase])
        obs[3+2*len(self.active_ids)+num_actions+2 : 3+2*len(self.active_ids)+num_actions+2+3] = np.array(self.target_loc)[:3]
        obs[3+2*len(self.active_ids)+num_actions+2+3 : 3+2*len(self.active_ids)+num_actions+2+3+1] = np.array([right_hand_state])
        obs[3+2*len(self.active_ids)+num_actions+2+3+1 : 3+2*len(self.active_ids)+num_actions+2+3+1+1] = np.array([right_hand_state_1])
        
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = self.action[:-2] * self.config["action_scale"] + self.default_angles[self.active_ids]
        if self.action[-1] < 0.5 :
            self.hands.grasp("right")
        else :
            self.hands.release("right")
        self.hands.publish_cmd()
        self.low_cmd.motor_cmd[29].q =  1
        for i in range(len(self.active_ids)):
            motor_idx = self.active_ids[i]
            self.low_cmd.motor_cmd[motor_idx].q = SMOOTHNESS_FACTOR*(target_dof_pos[i] - self.qj[i]) + self.qj[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        
        # send the command
        self.send_cmd(self.low_cmd)
        t2 = time.time()
        print(t2-self.curr_time)
        time_taken = t2 - t1
        if time_taken < self.config["control_dt"]:
            time.sleep(self.config["control_dt"]-time_taken)
        
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
    hands.release("right")
    controller = ControllerPickTop(config, hands, policy_name="Pick_Top_UB")

    controller.move_to_default_pos()
    pipeline = rs.pipeline()
    config = rs.config()
    # Configure streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    target_loc = None
    if obj != "none":
        target_loc = get_target_point(pipeline)
    print("Target location from vision:", target_loc)

    if target_loc is not None:
        controller.target_loc[0] = target_loc[0]
        controller.target_loc[1] = target_loc[1]
        controller.target_loc[2] = target_loc[2]

    controller.default_pos_state()

    controller.curr_time = time.time()
    controller_done = controller.run()
    controller.curr_time = time.time()
    while True:
        try:
            controller_done = controller.run()
            # Press the select key to exit
            if controller_done or controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    controller.turn_around()
    controller.move_back()

    controller.move_to_default_pos(mode_switch=True)
    
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
