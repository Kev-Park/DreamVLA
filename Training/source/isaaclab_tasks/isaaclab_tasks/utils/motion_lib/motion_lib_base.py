import glob
import os.path as osp
import numpy as np
import joblib
import torch
import pickle
from isaaclab_tasks.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from easydict import EasyDict
from loguru import logger
from rich.progress import track
import pytorch_kinematics as pk2
import tqdm

from isaac_utils.rotations import(
    slerp,
    quat_to_exp_map,
    quaternion_to_matrix
)

JointNamesOrder = [
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



JointNamesOrder_UB = [
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

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.tensor(tensor)

class MotionLibBase():
    def __init__(self, num_envs, device, motion_file, sim_fps=50):
        # self.m_cfg = motion_lib_cfg
        self._sim_fps = 50
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        self.load_data(motion_file)
        self.setup_constants(fix_height = False,  multi_thread = False)
        return
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['joints']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            # self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            # self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            # print(self._motion_data_load[0])
            # import pdb; pdb.set_trace()
            self._motion_data_list = []
            for i in range(len(self._motion_data_load)):
                # import pdb; pdb.set_trace()
                with open(self._motion_data_load[i], "rb") as file:
                    self._motion_data_list.append(pickle.load(file))
            # self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action = self._motion_actions[f0l]
        return action

    
    def get_motion_state(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 
        f1l = frame_idx1 

        
        rg_pos0 = self.transl[motion_ids, f0l, :]
        rg_pos1 = self.transl[motion_ids, f1l, :]

        quats0 = self.quats[motion_ids, f0l, :]
        quats1 = self.quats[motion_ids, f1l, :]

        dof_pos0 = self.dof_pos[motion_ids, f0l, :]
        dof_pos1 = self.dof_pos[motion_ids, f1l, :]

        local_keypts0 = self.local_keypts[motion_ids, f0l, :, :]
        local_keypts1 = self.local_keypts[motion_ids, f1l, :, :]

        global_keypts0 = self.global_keypts[motion_ids, f0l, :, :]
        global_keypts1 = self.global_keypts[motion_ids, f1l, :, :]

        is_closed = (f0l > self.switch_idxs[motion_ids])

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend) * rg_pos0 + blend * rg_pos1  # ZL: apply offset
        # print(quats0, quats1)
        quats = slerp(quats0, quats1, blend)
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        local_keypts = (1.0 - blend_exp) * local_keypts0 + blend_exp * local_keypts1
        global_keypts = (1.0 - blend_exp) * global_keypts0 + blend_exp * global_keypts1
        if self.offsets is not None:
            rg_pos = rg_pos + self.offsets[motion_ids]
            global_keypts = global_keypts + self.offsets[motion_ids].unsqueeze(1)
        return_dict = {}
        return_dict["root_pos"] = rg_pos.clone()
        return_dict["root_rot"] = quats.clone()
        return_dict["dof_pos"] = dof_pos.clone()
        return_dict["local_keypts"] = local_keypts.clone()
        return_dict["global_keypts"] = global_keypts.clone()
        return_dict["grab_pos"] = self.grab_pos[motion_ids].clone()
        return_dict["offsets"] = self.offsets[motion_ids].clone()
        return_dict["is_closed"] = is_closed
        return return_dict

    def get_keypts(self, joint_angles, joint_names, pk2_robot):
    
        q_dict = {name: joint_angles[:, i] for i, name in enumerate( joint_names ) }

        tf_dict = pk2_robot.forward_kinematics( q_dict )

        keypts = torch.zeros((len(joint_angles), len(tf_dict), 3))
        # print(len(tf_dict))
        # exit(0)
        cntr = 0 
        
        for name in tf_dict.keys():
            tf_val = tf_dict[name].get_matrix()  
            t = tf_val[:, :3, -1 ]
            # print(name, t)
            keypts[:, cntr , :] = t 
            cntr += 1 
        # print(keypts)
        return keypts


    def transform_keypts(self, keypts, quat, translation):
        """
        Transform keypoints using a quaternion and translation.
        Args:
            keypts: Tensor of shape (N, K, 3) where N is the number of samples, K is the number of keypoints.
            quat: Tensor of shape (N, 4) representing the quaternion.
            translation: Tensor of shape (N, 3) representing the translation.
        Returns:
            Transformed keypoints of shape (N, K, 3).
        """
        # Convert quaternion to rotation matrix (final shape will be (N, 3, 3))
        rot_matrix = quaternion_to_matrix(quat)
        # Ensure keypts is of shape (N, K, 3)
        if keypts.dim() == 2:
            keypts = keypts.unsqueeze(1)
        elif keypts.dim() != 3 or keypts.shape[-1] != 3:
            raise ValueError("keypts must be of shape (N, K, 3) or (N, 3)")
        
        # Apply rotation and translation
        transformed_keypts = torch.bmm(keypts, rot_matrix.transpose(2, 1)) + translation.unsqueeze(1)
        return transformed_keypts


    
    def calc_grab_pos_offset(self, quat, dof_pos, joint_names, pk2_robot, idx, offset_x=0.12, offset_y=0.07, offset_z=0.0):
        q_dict = {name: dof_pos[:, i] for i, name in enumerate( joint_names ) }
        tf_dict = pk2_robot.forward_kinematics( q_dict )

        tf_val = tf_dict['right_wrist_yaw_link'].get_matrix()
        R = tf_val[idx, :3, :3]
        rot_matrix = quaternion_to_matrix(quat[idx]) @ R
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]
        offset = offset_x * x_axis + offset_y * y_axis + offset_z * z_axis
        return offset
    
    def load_motions(self):
        self._motion_dt = []
        self._motion_lengths = []
        self._motion_num_frames = []
        self.transl = []
        self.quats = []
        self.dof_pos = []
        self.local_keypts = []
        self.global_keypts = []
        self.grab_pos = []
        self.offsets = []
        self.switch_idxs = []
        urdf_path = "HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
        pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())
        self.joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        for i in tqdm.tqdm(range(len(self._motion_data_list))):
            motion_file_data = self._motion_data_list[i]
            self.transl.append(to_torch(np.array(motion_file_data['global_pose'].translation())[:200]).clone())
            
            self.transl[-1][:,2] += 0.035
            offset = torch.zeros(3)
            
            self.quats.append(to_torch(np.array(motion_file_data['global_pose'].rotation().wxyz))[:200].clone())
            self.dof_pos.append(to_torch(np.array(motion_file_data['joints'])[:200]).clone())

            self.local_keypts.append(self.get_keypts(self.dof_pos[-1], self.joint_names, pk2_robot)[:])
            self.global_keypts.append(self.transform_keypts(self.local_keypts[-1], self.quats[-1], self.transl[-1]))
            idx_shift = 15
            
            if 'grab_pos' in motion_file_data or 'grab_pos_real' in motion_file_data:
                idx_shift = 2
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx']+idx_shift,-2, :].clone())
                self.grab_pos[-1] += self.calc_grab_pos_offset(self.quats[-1],self.dof_pos[-1], self.joint_names, pk2_robot, motion_file_data['grab_idx']+idx_shift)
            elif 'squat_pos' in motion_file_data or 'squat_pos_real' in motion_file_data:
                idx_shift = 0
                # import pdb; pdb.set_trace()
                motion_file_data['grab_idx'] = torch.argmin(self.global_keypts[-1][:,0,2])
                print(motion_file_data['grab_idx'])
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],0, :].clone())
            elif 'hit_pos' in motion_file_data or 'hit_pos_real' in motion_file_data:
                idx_shift = 0
                motion_file_data['grab_idx'] = torch.argmax(self.global_keypts[-1][:,-1,0])
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-2, :].clone())
                self.grab_pos[-1] += self.calc_grab_pos_offset(self.quats[-1],self.dof_pos[-1], self.joint_names, pk2_robot, motion_file_data['grab_idx'],offset_x=0.21, offset_y=0., offset_z=0.03)
                offset[1] = - self.grab_pos[-1][1]
            elif 'kick_pos' in motion_file_data or 'kick_pos_real' in motion_file_data:
                idx_shift = 0
                motion_file_data['grab_idx'] = torch.argmax(self.global_keypts[-1][:,13,0])
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],13, :].clone())
                # self.grab_pos[-1] += self.calc_grab_pos_offset(self.quats[-1],self.dof_pos[-1], self.joint_names, pk2_robot, motion_file_data['grab_idx'],offset_x=0., offset_y=0., offset_z=0.0)
                offset[1] = - self.grab_pos[-1][1]
            elif 'jump_pos' in motion_file_data or 'jump_pos_real' in motion_file_data:
                idx_shift = 0
                motion_file_data['grab_idx'] = torch.argmax(self.global_keypts[-1][:,16,2])
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],16,:].clone())
            elif 'punch_pos' in motion_file_data or 'punch_pos_real' in motion_file_data:
                idx_shift = 0
                motion_file_data['grab_idx'] = torch.argmax(self.global_keypts[-1][:,-1,0])
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-2, :].clone())
                self.grab_pos[-1] += self.calc_grab_pos_offset(self.quats[-1],self.dof_pos[-1], self.joint_names, pk2_robot, motion_file_data['grab_idx'],offset_x=0.14, offset_y=0., offset_z=0.0)
                # offset[1] = - self.grab_pos[-1][1]
            elif 'sit_pos' in motion_file_data:
                motion_file_data['grab_idx'] = 80
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],0, :].clone())
                offset[1] = - self.grab_pos[-1][1]
            elif 'bimanual_pos' in motion_file_data:
                # import pdb; pdb.set_trace()
                self.grab_pos.append((self.global_keypts[-1][motion_file_data['grab_idx'],-2, :].clone()+self.global_keypts[-1][motion_file_data['grab_idx'],-10, :].clone())/2.)
            elif 'ground_grab_pos' in motion_file_data:
                idx_shift = 0
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-11, :].clone())
            elif 'ground_grab_pos_new' in motion_file_data:
                idx_shift = 0
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-3, :].clone())
            elif 'pick_grab_pos' in motion_file_data:
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-3, :].clone())
                self.grab_pos[-1][0] += 0.2
                self.grab_pos[-1][1] += 0.07
                idx_shift = 0
            elif 'open_drawer_pos' in motion_file_data or 'open_door_pos' in motion_file_data:
                # motion_file_data['grab_idx'] = torch.argmax(self.global_keypts[-1][:,-1,0])
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-2, :].clone())
                self.grab_pos[-1] += self.calc_grab_pos_offset(self.quats[-1],self.dof_pos[-1], self.joint_names, pk2_robot, motion_file_data['grab_idx'],offset_x=0.05, offset_y=0., offset_z=0.0)

                idx_shift = 10
                # offset[1] = - self.grab_pos[-1][1]
            elif 'follow_pos' in motion_file_data:
                self.grab_pos.append(self.global_keypts[-1][motion_file_data['grab_idx'],-1, :].clone())
                idx_shift = 0
            self.switch_idxs.append(to_torch(np.array(motion_file_data['grab_idx']+idx_shift)).clone())
            
            offset[0] = 2.1 - self.grab_pos[-1][0]
            self._motion_dt.append(1./20.)
            
            if 'grab_pos_real' in motion_file_data:
                offset[0] = 0.
                self._motion_dt[-1] = 2.5/20.
            
            if 'hit_pos_real' in motion_file_data:
                offset[0] = 0.
                offset[1] = 0.
                self._motion_dt[-1] = 1.5/20.
            
            if 'punch_pos_real' in motion_file_data:
                offset[0] = 0.
                offset[1] = 0.
                self._motion_dt[-1] = 1.5/20.

            if 'kick_pos_real' in motion_file_data:
                offset[0] = 0.
                offset[1] = 0.
                self._motion_dt[-1] = 2./20.

            if 'squat_pos_real' in motion_file_data:
                offset[0] = 0.
                offset[1] = 0.
                self._motion_dt[-1] = 1./20.
            
            self.offsets.append(offset)
            
            motion_num_frames = len(self.transl[-1])
            self._motion_lengths.append(motion_num_frames * self._motion_dt[-1])
            self._motion_num_frames.append(motion_num_frames)
        self.switch_idxs = torch.tensor(self.switch_idxs).to(self._device).float()
        self.offsets = torch.stack(self.offsets, dim=0).to(self._device).float()
        self.grab_pos = torch.stack(self.grab_pos, dim=0).to(self._device).float()
        self.local_keypts = torch.stack(self.local_keypts, dim=0).to(self._device).float()
        self.global_keypts = torch.stack(self.global_keypts, dim=0).to(self._device).float()
        self.transl = torch.stack(self.transl, dim=0).to(self._device).float()
        self.quats = torch.stack(self.quats, dim=0).to(self._device).float()
        self.dof_pos = torch.stack(self.dof_pos, dim=0).to(self._device).float()
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device, dtype=torch.int32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._num_motions = len(self._motion_data_list)
        print(f"Loaded {self._num_motions} motions finally!")
        # import pdb; pdb.set_trace()

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():
            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            
            return trans, height_diff

    def load_motion_with_skeleton(self,
                                  motion_data_list,
                                  motion_len,dt=0.05):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file = motion_data_list[f]
            
            seq_len = curr_file['joints'].shape[0]
            start, end = 0, motion_len
            
            trans = to_torch(np.array(curr_file['global_pose'].translation())).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            

            B, J, N = pose_aa.shape

            if self.mesh_parsers is not None:
                # trans, trans_fix = MotionLibRobot.fix_trans_height(pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
                # add "action" to curr_motion
                res[f] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
        return res
    

    def num_motions(self):
        return self._num_motions


    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend


    def _get_num_bodies(self):
        return self.num_bodies


    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)