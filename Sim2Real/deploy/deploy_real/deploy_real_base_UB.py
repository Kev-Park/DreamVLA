from typing import Union, List, Dict
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

import yaml
import pyrealsense2 as rs
from grid_cortex_client.cortex_client import CortexClient
import cv2
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_


MODEL_ID = "owlv2"
LEGGED_GYM_ROOT_DIR = "../../"
HOST = 'localhost'
PORT = 9999
SMOOTHNESS_FACTOR = 1.0
LOC_X = -0.04
LOC_Y = 0.04
LOC_Z = 0.34
TARGET_LOC = [0.42, -0.067, 0.293, 1., 0., 0., 0.]

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


from common.command_helper import create_damping_cmd, create_zero_cmd
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap



from dataclasses import dataclass, field

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmds_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorCmds_"):
    cmds: types.sequence["unitree_sdk2py.idl.unitree_go.msg.dds_.MotorCmd_"] = field(
        default_factory=lambda: []
    )


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorStates_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorStates_"):
    states: types.sequence["unitree_sdk2py.idl.unitree_go.msg.dds_.MotorState_"] = (
        field(default_factory=lambda: [])
    )


class UnitreeInspire:
    def __init__(self):  # Default to eth0

        self.joint_index = {
            "RightPinky": 0,
            "RightRing": 1,
            "RightMiddle": 2,
            "RightIndex": 3,
            "RightThumbBend": 4,
            "RightThumbRotation": 5,
            "LeftPinky": 6,
            "LeftRing": 7,
            "LeftMiddle": 8,
            "LeftIndex": 9,
            "LeftThumbBend": 10,
            "LeftThumbRotation": 11,
        }
        # Initialize with the list of motor commands directly
        self.cmd_publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.cmd_publisher.Init()
        self.state = None
        self.state_subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self.state_subscriber.Init(self._state_cb, 1)
        self.finger_cmd = MotorCmds_(
            cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(len(self.joint_index))]
        )

    def _state_cb(self, msg):
        self.state = msg

    def moveFingersToAngle(
        self,
        joint_positions: Union[List[float], Dict[str, float]],
        left_angles: Union[List[float], None] = None,
    ):
        """
        Move fingers to specified angles.

        Args:
            joint_positions: Either a list of positions for right hand joints or a dictionary mapping joint names to positions
            left_angles: Optional list of positions for left hand joints when joint_positions is a list for right hand
        """
        # Convert input to dictionary format if it's a list
        if isinstance(joint_positions, list):
            if left_angles is not None:
                if len(left_angles) != len(self.joint_index) // 2:
                    raise ValueError(
                        f"Left angles must be a list of length {len(self.joint_index) // 2} with the order of {list(self.joint_index.keys())[: len(self.joint_index) // 2]}"
                    )
                if len(joint_positions) != len(self.joint_index) // 2:
                    raise ValueError(
                        f"Joint positions must be a list of length {len(self.joint_index) // 2} with the order of {list(self.joint_index.keys())[len(self.joint_index) // 2 :]}"
                    )
                joint_positions += left_angles
            if len(joint_positions) != len(self.joint_index):
                raise ValueError(
                    f"Joint positions must be a list of length {len(self.joint_index)} with the order of {list(self.joint_index.keys())}"
                )
            joint_positions = dict(zip(self.joint_index.keys(), joint_positions))

        # Apply joint positions to finger command
        for joint_name, joint_position in joint_positions.items():
            if joint_name not in self.joint_index:
                raise ValueError(f"Unknown joint name: {joint_name}")
            index = self.joint_index[joint_name]
            self.finger_cmd.cmds[index].q = np.clip(joint_position, 0.0, 1.0)

        # Send the command
        self.cmd_publisher.Write(self.finger_cmd)

    def getFingerAngles(self, return_dict: bool = False):
        if self.state is None:
            return {} if return_dict else None

        angles = self.state.states
        if return_dict:
            angles_dict = {}
            for joint_name, joint_index in self.joint_index.items():
                angles_dict[joint_name] = angles[joint_index].q
            return angles_dict
        else:
            return angles

    def publish_cmd(self):
        self.cmd_publisher.Write(self.finger_cmd)

    def grasp(self, hand: str = "", value: float = 0.0):
        if hand == "left" or hand == "":
            for joint_name in [
                "LeftThumbBend",
                "LeftIndex",
                "LeftMiddle",
                "LeftRing",
                "LeftPinky",
                "LeftThumbRotation",
            ]:
                self.finger_cmd.cmds[self.joint_index[joint_name]].q = CLOSED_CONFIG[joint_name]
        if hand == "right" or hand == "":
            for joint_name in [
                "RightThumbBend",
                "RightIndex",
                "RightMiddle",
                "RightRing",
                "RightPinky",
                "RightThumbRotation",
            ]:
                self.finger_cmd.cmds[self.joint_index[joint_name]].q = CLOSED_CONFIG[joint_name]
        
    def release(self, hand: str = ""):
        if hand == "left" or hand == "":
            for joint_name in [
                "LeftThumbBend",
                "LeftIndex",
                "LeftMiddle",
                "LeftRing",
                "LeftPinky",
                "LeftThumbRotation",
            ]:
                self.finger_cmd.cmds[self.joint_index[joint_name]].q = OPEN_CONFIG[joint_name]
        if hand == "right" or hand == "":
            for joint_name in [
                "RightThumbBend",
                "RightIndex",
                "RightMiddle",
                "RightRing",
                "RightPinky",
                "RightThumbRotation",
            ]:
                self.finger_cmd.cmds[self.joint_index[joint_name]].q = OPEN_CONFIG[joint_name]
        

def get_box_center(box, image_size):
    """
    Calculate the center of a single bounding box given in xyxy format.

    Args:
        box (list or np.ndarray): A bounding box with format [x_min, y_min, x_max, y_max].

    Returns:
        tuple: Center point (x_center, y_center).
    """
    x_min, y_min, x_max, y_max = box
    print((image_size[0] * image_size[1]) / (x_max - x_min) * (y_max - y_min))
    if image_size[0] * image_size[1] * 0.4 < (x_max - x_min) * (y_max - y_min):
        return -1, -1
    print(
        "proportion: ",
        (((x_max - x_min) * (y_max - y_min)) / (image_size[0] * image_size[1])),
    )
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center



class Controller:
    def __init__(self, config, hands, policy_name="Pick_UB") -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.hands = hands
        self.curr_time = None
        self.phase = 0
        self.added = False
        self.subtracted = False
        self.robot_joint_names = [
                "left_hip_pitch_joint",
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
                "waist_roll_joint",
                "waist_pitch_joint",
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
                "right_wrist_yaw_joint",
            ]

        active_joints = ["waist_yaw_joint",
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
        
        self.active_ids = [self.robot_joint_names.index(name) for name in active_joints]
        self.robot_joint_ids = [config["joint_names"].index(name) for name in self.robot_joint_names]
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)[self.robot_joint_ids]
        self.kps = np.array(config["kps"], dtype=np.float32)[self.robot_joint_ids]
        self.kds = np.array(config["kds"], dtype=np.float32)[self.robot_joint_ids]
        self.target_loc = TARGET_LOC
        self.target_quat = [1, 0, 0, 0]
        self.action = np.zeros(config["num_actions"], dtype=np.float32)

        # Initialize the policy network
        self.policy = torch.jit.load(config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR) + policy_name + ".pt")
        
        # Initializing process variables
        self.obs = np.zeros(config["num_obs"], dtype=np.float32)
        self.cmd = np.array([0, 0, 0])
        self.counter = 0

        if config["msg_type"] == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            
            self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmdHG)
            self.arm_sdk_publisher.Init()

            self.lowstate_subscriber = ChannelSubscriber(config["lowstate_topic"], LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()
        self.qj = np.zeros(config["num_actions"]-2, dtype=np.float32)
        self.dqj = np.zeros(config["num_actions"]-2, dtype=np.float32)

        
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.arm_sdk_publisher.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config["control_dt"])
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            self.low_cmd.motor_cmd[29].q =  1
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config["control_dt"])

    def move_to_default_pos(self, mode_switch=False):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config["control_dt"])

        default_pos = np.array(self.config["default_angles"])[self.robot_joint_ids]
        kps = np.array(self.config["kps"])[self.robot_joint_ids]
        kds = np.array(self.config["kds"])[self.robot_joint_ids]
        dof_size = len(self.robot_joint_ids)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(12,dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].q
        

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            self.low_cmd.motor_cmd[29].q =  1
            if mode_switch:
                self.low_cmd.motor_cmd[29].q = 1. - alpha
            # print(self.low_cmd)
            for j in range(12,dof_size):
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = kps[j]
                self.low_cmd.motor_cmd[j].kd = kds[j]
                self.low_cmd.motor_cmd[j].tau = 0
            
            self.send_cmd(self.low_cmd)
            time.sleep(self.config["control_dt"])

    def turn_around(self):
        print("Turning around.")
        # move time 2s
        total_time = 3
        num_step = int(total_time / self.config["control_dt"])
        # default_pos = np.array(self.config["default_angles"])[self.robot_joint_ids]
        kps = np.array(self.config["kps"])[self.robot_joint_ids]
        kds = np.array(self.config["kds"])[self.robot_joint_ids]
        dof_size = len(self.robot_joint_ids)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        new_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(12,dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].q
            new_dof_pos[i] = self.low_state.motor_state[i].q
            if i==22:
                new_dof_pos[i] -= 0.4
            if i==12:
                new_dof_pos[i] -= 1.3
            if i==25:
                new_dof_pos[i] -= 1.
            
        print(init_dof_pos, new_dof_pos)
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            self.low_cmd.motor_cmd[29].q =  1
            # print(self.low_cmd)
            for j in range(12,dof_size):
                target_pos = new_dof_pos[j]
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = kps[j]
                self.low_cmd.motor_cmd[j].kd = kds[j]
                self.low_cmd.motor_cmd[j].tau = 0
            
            self.send_cmd(self.low_cmd)
            time.sleep(self.config["control_dt"])

    def move_back(self):
        print("Moving back.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config["control_dt"])
        self.hands.release("right")
        self.hands.publish_cmd()
        # default_pos = np.array(self.config["default_angles"])[self.robot_joint_ids]
        kps = np.array(self.config["kps"])[self.robot_joint_ids]
        kds = np.array(self.config["kds"])[self.robot_joint_ids]
        dof_size = len(self.robot_joint_ids)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        new_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(12,dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].q
            new_dof_pos[i] = self.low_state.motor_state[i].q
            if i==22:
                new_dof_pos[i] += 1.4
            if i==25:
                new_dof_pos[i] -= 0.
            if i==27:
                new_dof_pos[i] -= 1.1
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            self.low_cmd.motor_cmd[29].q =  1
            # print(self.low_cmd)
            for j in range(12,dof_size):
                target_pos = new_dof_pos[j]
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = kps[j]
                self.low_cmd.motor_cmd[j].kd = kds[j]
                self.low_cmd.motor_cmd[j].tau = 0
            
            self.send_cmd(self.low_cmd)
            time.sleep(self.config["control_dt"])
                
    
    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self.low_cmd.motor_cmd[29].q =  1
            default_pos = np.array(self.config["default_angles"])[self.robot_joint_ids]
            kps = np.array(self.config["kps"])[self.robot_joint_ids]
            kds = np.array(self.config["kds"])[self.robot_joint_ids]
            for i in range(len(self.robot_joint_ids)):
                self.low_cmd.motor_cmd[i].q = default_pos[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config["control_dt"])

    def run(self):
        raise NotImplementedError

