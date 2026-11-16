import sys as _reference_sys
from pathlib import Path as _ReferencePath

# Keep direct script invocation working without changing the installed environment.
_reference_sys.path.insert(0, str(_ReferencePath(__file__).resolve().parents[2] / "holosoma_adapters"))
from kinematics_autograd import autograd_forward_kinematics

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import pytorch_kinematics as pk
import numpy as np
import pickle as pkl
import viser
from viser.extras import ViserUrdf
import jax.numpy as jnp
import jaxlie
import yourdfpy
import pyroki as pk2
import pickle
import time
import trimesh

import torch.optim as optim
from isaac_utils.rotations import(
    quat_conjugate,
    quaternion_to_matrix,
    slerp
)
import tqdm
# =======================
# Select device here
# =======================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# =======================

# Load URDF and create kinematic chain
urdf_path = '../../../HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
chain = pk.build_chain_from_urdf(open(urdf_path).read())
chain = chain.to(device=DEVICE)
_forward_kinematics = autograd_forward_kinematics(chain)
pkl_path = 'base.pkl'
SMOOTH_AMT = 20
PAUSE_AMT = 10
INTERP_AMT = 10
SAVE_DIR = "../Punch_real2/"
OFFSET_X = -0.23
WRIST_TO_COLLISION = 0.3
NX = 5
NY = 10
NZ = 25
X_MIN = -0.05
X_MAX = 0.
Y_MIN = -0.1
Y_MAX = 0.1
Z_MIN = -0.15
Z_MAX = 0.1
SAME_LEFT_ARM_TRAJ = True
# Generate normalized grid
x = np.linspace(X_MIN, X_MAX, NX)
y = np.linspace(Y_MIN, Y_MAX, NY)
z = np.linspace(Z_MIN, Z_MAX, NZ)
SHIFT_Xs, SHIFT_Ys, SHIFT_Zs = np.meshgrid(x, y, z)
SHIFT_Xs = SHIFT_Xs.flatten()
SHIFT_Ys = SHIFT_Ys.flatten()
SHIFT_Zs = SHIFT_Zs.flatten()

VIS = True
START_FROM = 1
num_timesteps = 196

heightmap = np.zeros((1000, 1000), dtype=np.float32)  # Dummy heightmap for visualization
heightmap = pk2.collision.Heightmap(
    pose=jaxlie.SE3.identity(),
    size=jnp.array([0.01, 0.01, 1.0]),
    height_data=heightmap,
)

if VIS :
    urdf = yourdfpy.URDF.load('../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    base_frame_new = server.scene.add_frame("/base_new", show_axes=False)
    urdf_vis_new = ViserUrdf(server, urdf, root_node_name="/base_new")
    playing = server.gui.add_checkbox("playing", False)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 2 + INTERP_AMT + PAUSE_AMT, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

def compute_cost(joint_angles, trans, quats, ref_dists):
    # L2 cost to target
    l2_cost = 0.*torch.nn.functional.mse_loss(joint_angles, target_joint_angles[:, active_joint_ids])
    q_dict = {name: joint_angles[:, i] for i, name in enumerate( active_joint_names ) }
    for id in inactive_joint_ids:
        name = joint_names[id]
        q_dict[name] = target_joint_angles[:, id]
    fk_results = _forward_kinematics(q_dict)
    cost2 = torch.zeros(joint_angles.shape[0], device=DEVICE)
    rot_matrix = quaternion_to_matrix(quats)
    i = 0
    for link_name, tf in fk_results.items():
        if i == 36:
            pos = tf.get_matrix()[:,:3,3]
            pos_ref = fk_results_ref[link_name].get_matrix()[:,:3,3]
            cost2 += 0.*torch.mean(torch.norm(pos[1:] - pos[:-1], dim=1))
            transformed_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
            transformed_keypts_ref = torch.bmm(pos_ref.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
            transformed_keypts_ref[:, 0] += SHIFT_X
            transformed_keypts_ref[:, 1] += SHIFT_Y
            transformed_keypts_ref[:, 2] += SHIFT_Z
            rel_dists = torch.norm(transformed_keypts[1:] - transformed_keypts[:-1], dim=1)
            vals = rel_dists - ref_dists
            cost2[1:-1] += torch.abs(vals[1:] - vals[:-1])
            cost2[1:] += torch.abs(vals)
            cost2[1:] += 10*vals**2
            cost2[0] += torch.norm(transformed_keypts[0] - transformed_keypts_ref[0], p=2)
            cost2[grab_idx-5:grab_idx+5] += torch.norm(transformed_keypts[grab_idx-5:grab_idx+5] - transformed_keypts_ref[grab_idx-5:grab_idx+5], dim=1, p=2)
            # print(transformed_keypts_ref[grab_idx:])
        elif i == 38:
            rot_mat = tf.get_matrix()[:,:3,:3]
            rot_mat = torch.bmm(rot_matrix, rot_mat)
            rot_mat_ref = torch.tensor([[1, 0, 0], 
                                        [0, 1, 0], 
                                        [0, 0, 1]], dtype=torch.float32, device=DEVICE).T
            rot_mat = torch.bmm(rot_mat, rot_mat_ref.unsqueeze(0).expand(rot_mat.shape[0], -1, -1))
            angle = torch.acos(torch.clamp((rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2] - 1) / 2, -0.999, .999))
            cost2 += 0.*angle
        else:
            i += 1
            continue
        i += 1
    return l2_cost + torch.mean(cost2)

joint_angles_first_col = []
target_positions = []
joint_angles_next = []
motion_data_base = pkl.load(open(pkl_path, 'rb'))
for i in tqdm.tqdm(range(len(SHIFT_Xs))):
    SHIFT_X = SHIFT_Xs[i]
    SHIFT_Y = SHIFT_Ys[i]
    SHIFT_Z = SHIFT_Zs[i]
    pkl_path = str(i)+'.pkl'
    motion_data = pkl.load(open(pkl_path, 'rb'))
    joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    stand_still_joint_angles = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0., 0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.]
    
    # All tensors to DEVICE:
    target_trans = torch.tensor(np.array(motion_data['global_position']), device=DEVICE).clone()[START_FROM:]
    target_quats = torch.tensor(np.array(motion_data['global_pose'].rotation().wxyz), device=DEVICE).clone()[START_FROM:]
    target_trans *= 0.
    
    target_quats *= 0.
    target_quats[:,0] = 1.
    target_joint_angles_base = torch.tensor(np.array(motion_data_base['joints']), device=DEVICE).clone()[START_FROM:]
    target_joint_angles = torch.tensor(np.array(motion_data['joints']), device=DEVICE).clone()[START_FROM:]
    if SAME_LEFT_ARM_TRAJ:
        target_joint_angles[:, 13:20] = target_joint_angles_base[:, 13:20]
    
    target_joint_angles[:, :12] = torch.tensor(stand_still_joint_angles, device=DEVICE).unsqueeze(0)[:,:12]
    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate( joint_names ) }
    rot_matrix = quaternion_to_matrix(target_quats)
    fk_results = _forward_kinematics(q_dict)
    pos_right_ankle = fk_results["right_ankle_roll_link"].get_matrix()[:,:3,3]
    pos_right_ankle = torch.bmm(pos_right_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + target_trans
    target_trans[:,2] -= pos_right_ankle[:,2]

    grab_idx = 27

    inactive_joint_names = ['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
    active_joint_ids = [i for i, name in enumerate(joint_names) if name not in inactive_joint_names]
    active_joint_names = [name for name in joint_names if name not in inactive_joint_names]
    inactive_joint_ids = [i for i, name in enumerate(joint_names) if name in inactive_joint_names]
    q_dict = {name: target_joint_angles_base[:, i] for i, name in enumerate( joint_names ) }
    fk_results_ref = _forward_kinematics(q_dict)
    target_trans[:,:] = target_trans[:1,:]
    target_quats[:,:] = target_quats[:1,:]
    joint_angles = torch.nn.Parameter(torch.tensor(target_joint_angles[:, active_joint_ids], device=DEVICE).clone())
    
    trans = torch.tensor(target_trans, device=DEVICE)
    quats = torch.tensor(target_quats, device=DEVICE)
    target_positions.append([SHIFT_X, SHIFT_Y, SHIFT_Z])
    optimizer = optim.Adam([joint_angles], lr=0.003)

    q_dict = {name: target_joint_angles_base[:, i] for i, name in enumerate( joint_names ) }
    fk_results = _forward_kinematics(q_dict)
    tf = fk_results["right_wrist_pitch_link"]
    pos = tf.get_matrix()[:,:3,3]
    rot_matrix = quaternion_to_matrix(quats)
    wrist_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
    ref_dists = torch.norm(wrist_keypts[1:] - wrist_keypts[:-1], dim=1)

    for j in range(2000):
        optimizer.zero_grad()
        if j == 399 :
            debug = True
        else:
            debug = False
        t1 = time.time()
        cost = compute_cost(joint_angles, trans, quats, ref_dists)
        t2 = time.time()
        t1 = time.time()
        cost.backward()
        t2 = time.time()
        if cost.item() < 0.001 :
            break
        optimizer.step()
        if j % 100 == 0:
            print(f"Step {j}, Cost: {cost.item()}")

    joint_angles_next.append(joint_angles.data.cpu().numpy())
    joints_new_ = target_joint_angles.clone().to(device=DEVICE)
    joints_new_[:,active_joint_ids] = joint_angles.data
    num_timesteps = target_joint_angles.shape[0]
    
    joints_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, joints_new_.shape[1]), dtype=joints_new_.dtype, device=DEVICE)
    trans_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, 3), dtype=joints_new_.dtype, device=DEVICE)
    quats_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, 4), dtype=joints_new_.dtype, device=DEVICE)
    trans_new[INTERP_AMT + PAUSE_AMT:, :] = target_trans.clone()
    quats_new[INTERP_AMT + PAUSE_AMT:, :] = target_quats.clone()
    joints_new[INTERP_AMT + PAUSE_AMT:, :] = joints_new_.clone()
    quats_new[:INTERP_AMT + PAUSE_AMT, 0] = 1.0

    joints_new[:PAUSE_AMT, :] = torch.tensor(stand_still_joint_angles, device=DEVICE).unsqueeze(0).repeat(PAUSE_AMT, 1)
    joints_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = joints_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
        (joints_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - joints_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
        torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)

    
    first_viol_i = 0
    q_dict = {name: joints_new[:, i] for i, name in enumerate( joint_names ) }
    rot_matrix = quaternion_to_matrix(quats_new)
    fk_results = _forward_kinematics(q_dict)
    pos_right_ankle = fk_results["right_ankle_roll_link"].get_matrix()[:,:3,3]
    pos_right_ankle = torch.bmm(pos_right_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_new
    pos_left_ankle = fk_results["left_ankle_roll_link"].get_matrix()[:,:3,3]
    pos_left_ankle = torch.bmm(pos_left_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_new

    if pos_right_ankle[PAUSE_AMT+INTERP_AMT,0] < pos_left_ankle[PAUSE_AMT+INTERP_AMT,0]:
        trans_new[:PAUSE_AMT, 2] -= pos_right_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = pos_right_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_right_ankle[:PAUSE_AMT, :2]
        trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
            (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
            torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)
    else:
        trans_new[:PAUSE_AMT, 2] -= pos_left_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = pos_left_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_left_ankle[:PAUSE_AMT, :2]
        trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
            (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
            torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)

    quats_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = slerp(
        quats_new[PAUSE_AMT-1:PAUSE_AMT, :],
        quats_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :],
        torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)
    )
    trans_new[80:] = trans_new[79:80]
    quats_new[80:] = quats_new[79:80]
    joints_new[80:] = joints_new[79:80]
    Ts_world_root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.array(quats_new.cpu())), jnp.array(trans_new.cpu())
    )

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    print(motion_data_base["grab_idx"])
    with open(SAVE_DIR + pkl_path[:-4] + "_" + str(int(SHIFT_X*1000)) + "_" + str(int(SHIFT_Y*1000)) + "_" + str(int(SHIFT_Z*1000)) + ".pkl","wb") as f:
        pickle.dump({"global_pose": Ts_world_root , "joints": joints_new.cpu(), "global_position": trans_new.cpu(), "punch_pos_real": motion_data["punch_pos"], "grab_idx": motion_data["grab_idx"]+(PAUSE_AMT+INTERP_AMT-START_FROM)}, f)

if VIS :
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                time.sleep(0.1)
            tstep = timestep_slider.value
            base_frame_new.wxyz = np.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame_new.position = np.array(Ts_world_root.wxyz_xyz[tstep][4:]) + np.array([0, 0, 0.035])
            urdf_vis_new.update_cfg(np.array(joints_new[tstep].cpu()))
