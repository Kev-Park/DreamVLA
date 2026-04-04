import time

import numpy as np
import torch

SMOOTHNESS_FACTOR = 1.0
POLICY_TIME = 4.0
BUTTON_PRESS_TIME = 3.0


class Controller:
    def __init__(self, agent, policy_path: str) -> None:
        self.curr_time = None
        self.phase = 0
        self.added = False
        self.subtracted = False
        # Initialize the policy network
        self.policy = torch.jit.load(policy_path)
        num_actions = 7
        num_obs = 28
        # Initializing process variables
        self.qj = np.zeros(num_actions, dtype=np.float32)
        self.dqj = np.zeros(num_actions, dtype=np.float32)
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_dof_pos = [0.35, 0.16, 0, 0.87, 0, 0, 0]
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.cmd = np.array([0, 0, 0])
        self.counter = 0
        self.agent = agent

    def run(self):
        self.counter += 1
        low_state = self.agent.getLowState()
        active_idx = self.agent.getLeftArmIndices()
        print(active_idx)
        # Get the current joint position and velocity
        for i in range(len(active_idx)):
            self.qj[i] = low_state.motor_state[active_idx[i]].q
            self.dqj[i] = low_state.motor_state[active_idx[i]].dq
        # create observation
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        default_angles = [0.35, 0.16, 0, 0.87, 0, 0, 0]
        print(len(self.qj), len(default_angles))
        qj_obs = qj_obs - default_angles
        dqj_obs = dqj_obs
        return_val = False
        if (time.time() - self.curr_time) > POLICY_TIME + 2 * BUTTON_PRESS_TIME:
            return_val = True
        
        self.obs[:3] = self.target_loc + np.array([-0.05, 0., 0.])*np.cos(self.target_yaw) - np.array([0., -0.05, 0.])*np.sin(self.target_yaw) 
        if (time.time() - self.curr_time) > POLICY_TIME and (time.time() - self.curr_time) < POLICY_TIME + BUTTON_PRESS_TIME:
            self.obs[:3] = self.target_loc + np.array([0.08, 0., 0.])*np.cos(self.target_yaw) - np.array([0., 0.05, 0.])*np.sin(self.target_yaw)
        
        if (time.time() - self.curr_time) > POLICY_TIME + 2 * BUTTON_PRESS_TIME:
            return_val = True

        num_actions =7
        print(self.target_loc)
        self.obs[3:7] = [1, 0, 0, 0]
        self.obs[7 : 7 + num_actions] = qj_obs
        self.obs[7 + num_actions : 7 + num_actions * 2] = dqj_obs
        self.obs[7 + num_actions * 2 : 7 + num_actions * 3] = self.action

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = default_angles + self.action * 0.5

        # Build low cmd
        angles = {}
        for i in range(len(active_idx)):
            motor_idx = active_idx[i]
            angles[motor_idx] = (
                SMOOTHNESS_FACTOR * (target_dof_pos[i] - self.qj[i]) + self.qj[i]
            )
        angles[12] = 0.0
        angles[13] = 0.0
        angles[14] = 0.0
        self.agent.setArmAngles(angles, kd_scale=10, smooth=False)
        time.sleep(0.02)
        return return_val

    def press_button(self, target_pos):
        # self.agent.moveToDefaultLeftArmState(smooth=True)
        self.curr_time = time.time()

        self.target_loc = np.array(target_pos[:3])
        self.target_yaw = target_pos[3]
        print("!!!!!Target yaw: ", np.rad2deg(self.target_yaw), "!!!!!!!!!!!!!!")
        print("Target location from vision:", self.target_loc)
        if self.target_loc is not None:
            while True:
                try:
                    # self.agent.moveToDefaultLeftArmState(repeat=True, smooth=True)
                    controller_done = self.run()
                    # Press the select key to exit
                    if controller_done:
                        break
                except KeyboardInterrupt:
                    break
        
            self.curr_time = time.time()
            angles = {}
            active_idx = self.agent.getLeftArmIndices()
            default_angles = [0.35, 0.16, 0, 0.87, 0, 0, 0]
            for i in range(len(active_idx)):
                motor_idx = active_idx[i]
                angles[motor_idx] = (
                    default_angles[i]
                )
            angles[13] = 0.0
            angles[14] = 0.0
            self.agent.setArmAngles(angles, kd_scale=10, smooth=True)
            self.agent.disableArms()
            self.agent.locomotion.sport_client.SetFsmId(4)
            time.sleep(2)
            print("Done")
        # self.agent.moveToDefaultLeftArmState(smooth=True)


def press_button(agent, target_pos, policy_path):
    controller = Controller(agent, policy_path)
    controller.press_button(target_pos)

