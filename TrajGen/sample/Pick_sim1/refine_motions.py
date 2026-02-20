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
import glob
import torch.optim as optim
from isaac_utils.rotations import(
    quat_conjugate,
    quaternion_to_matrix,
    slerp
)

# Load URDF and create kinematic chain
urdf_path = '../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
chain = pk.build_chain_from_urdf(open(urdf_path).read())
pkl_paths = '*.pkl'
pkl_paths = glob.glob(pkl_paths)
SMOOTH_AMT = 20
PAUSE_AMT = 10
INTERP_AMT = 10
WRIST_TO_COLLISION = 0.35
VISUALIZE = True
OFFSET_Z = 0.86
OFFSET_X = -0.35
SAVE_DIR = "../Pick_sim2/"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def compute_cost(joint_angles, trans, quats, offset_x=OFFSET_X, offset_z=OFFSET_Z, debug=False):
    # L2 cost to target
    l2_cost = 0.*torch.nn.functional.mse_loss(joint_angles, target_joint_angles[:, active_joint_ids])
    
    # Forward kinematics to get link positions
    q_dict = {name: joint_angles[:, i] for i, name in enumerate( active_joint_names ) }
    for id in inactive_joint_ids:
        name = joint_names[id]
        q_dict[name] = target_joint_angles[:, id]
    fk_results = chain.forward_kinematics(q_dict)
    # Collect all link positions
    cost2 = torch.zeros(joint_angles.shape[0])
    rot_matrix = quaternion_to_matrix(quats)
    i = 0
    for link_name, tf in fk_results.items():
        if i == 36:
            pos = tf.get_matrix()[:,:3,3]
            pos_ref = fk_results_ref[link_name].get_matrix()[:,:3,3]

            cost2 += 0.*torch.mean(torch.norm(pos[1:] - pos[:-1], dim=1))
            transformed_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
            transformed_keypts_ref = torch.bmm(pos_ref.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
            rel_dists = torch.norm(transformed_keypts[1:] - transformed_keypts[:-1], dim=1)
            
            # Penalize large jumps in joint angles between frames
            max_ref_dists = torch.max(ref_dists)
            ja_diff = 0.03*torch.sum(torch.abs(joint_angles[1:] - joint_angles[:-1]), dim=1)
            cost2[:-1] += ja_diff * (1.2-ref_dists/max_ref_dists)

            # Penalize wrist speed deviation from reference retargeted motion
            vals = rel_dists - ref_dists
            cost2[1:-1] += 10*(vals[1:] - vals[:-1])**2
            cost2[1:] += torch.abs(vals) # penalize high speed
            cost2[1:] += 10*vals**2 # penalize speed again

            # penalize right shoulder being greater than -0.15, encourage to stay below
            cost2[:] += 10.*(joint_angles[:, -6]+0.15)*(joint_angles[:, -6]>-0.15) 

            # penalize first frame wrist position to match last frame
            cost2[0] += torch.norm(transformed_keypts[0] - transformed_keypts_ref[0], p=2)


            # Moved wrist collision penalty to hand


            # penalize wrist height to remain above table
            transformed_keypts_ref[grab_idx:, 2] = torch.maximum(transformed_keypts_ref[grab_idx:, 2], torch.tensor(offset_z))
            cost2[grab_idx:] += torch.norm(transformed_keypts[grab_idx:] - transformed_keypts_ref[grab_idx:], p=2)

            # add "laziness" penalty — linearly decays from 1.0 at frame 0 to 0.0 at grab_idx
            rest_pose = torch.tensor(init_joint_angles)[active_joint_ids]
            laziness_weight = torch.linspace(1.0, 0.0, grab_idx)
            cost2[:grab_idx] += laziness_weight * torch.sum(torch.abs(joint_angles[:grab_idx] - rest_pose), dim=1)

        elif i == 38:
            rot_mat = tf.get_matrix()[:,:3,:3]
            rot_mat = torch.bmm(rot_matrix, rot_mat)
            rot_mat_ref = torch.tensor([[1, 0, 0], 
                                        [0, 1, 0], 
                                        [0, 0, 1]], dtype=torch.float32, device=DEVICE).T
            rot_mat = torch.bmm(rot_mat, rot_mat_ref.unsqueeze(0).expand(rot_mat.shape[0], -1, -1))
            angle = torch.acos(torch.clamp((rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2] - 1) / 2, -0.999, .999))
            cost2 += 0.3*angle



            # penalize hand tip colliding with table
            HAND_TIP_OFFSET = 0.30
            hand_tf = tf.get_matrix()                   # (N, 4, 4)
            hand_pos = hand_tf[:, :3, 3]                # joint origin (N, 3)
            hand_rot = hand_tf[:, :3, :3]               # hand local rotation (N, 3, 3)
            local_tip = torch.tensor([HAND_TIP_OFFSET, 0., 0.], device=DEVICE)
            tip_pos = hand_pos + torch.bmm(hand_rot, local_tip.view(1, 3, 1).expand(hand_rot.shape[0], -1, -1)).squeeze(-1)
            transformed_tip = torch.bmm(tip_pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans

            def table_collision_cost(pts, gi):
                return 2 * (pts[:gi, 0] > grab_pos[0] + offset_x) * \
                    (pts[:gi, 2] < offset_z) * \
                    (torch.minimum(- grab_pos[0] - offset_x + pts[:gi, 0], - pts[:gi, 2] + offset_z))**2

            # per-frame check
            cost2[:grab_idx] += table_collision_cost(transformed_tip, grab_idx)

            # midpoint interpolation check to prevent tunneling
            tip_mid = (transformed_tip[:-1] + transformed_tip[1:]) / 2
            grab_idx_mid = max(grab_idx - 1, 0)
            cost2[:grab_idx_mid] += table_collision_cost(tip_mid, grab_idx_mid)
        else:
            i += 1
            continue
        i += 1

    return l2_cost + torch.mean(cost2)

if VISUALIZE:
    urdf = yourdfpy.URDF.load('../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    base_frame_new = server.scene.add_frame("/base_new", show_axes=False)
    urdf_vis_new = ViserUrdf(server, urdf, root_node_name="/base_new")
    playing = server.gui.add_checkbox("playing", False)
    timestep_slider = server.gui.add_slider("timestep", 0, 195 + INTERP_AMT + PAUSE_AMT, 1, 0)
    heightmap = np.zeros((1000, 1000), dtype=np.float32)  # Dummy heightmap for visualization
    heightmap = pk2.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.01, 1.0]),
        height_data=heightmap,
    )

    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

for pkl_path in pkl_paths:

    motion_data = pkl.load(open(pkl_path, 'rb'))

    target_trans = torch.tensor(np.array(motion_data['global_position'])).clone()
    target_quats = torch.tensor(np.array(motion_data['global_pose'].rotation().wxyz)).clone()

    # Target joint angles (example, replace with your actual target)
    target_joint_angles = torch.tensor(np.array(motion_data['joints'])).clone()
    grab_pos = torch.tensor(np.array(motion_data['grab_pos'])).clone()

    joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    init_joint_angles = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0., 0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.]
    inactive_joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
    active_joint_ids = [i for i, name in enumerate(joint_names) if name not in inactive_joint_names]
    active_joint_names = [name for name in joint_names if name not in inactive_joint_names]
    inactive_joint_ids = [i for i, name in enumerate(joint_names) if name in inactive_joint_names]
    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate( joint_names ) }
    fk_results_ref = chain.forward_kinematics(q_dict)
    # copy a tensor
    # Initial guess for joint angles (can be zeros or random)
    joint_angles = torch.nn.Parameter(torch.tensor(target_joint_angles[:, active_joint_ids]).clone())  # Only optimize active joints
    trans = torch.tensor(target_trans) # Translation offset
    quats = torch.tensor(target_quats) # Quaternion offset
    optimizer = optim.Adam([joint_angles], lr=0.001)
    grab_idx = motion_data["grab_idx"]
    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate( joint_names ) }
    fk_results = chain.forward_kinematics(q_dict)
    tf = fk_results["right_wrist_pitch_link"]
    pos = tf.get_matrix()[:,:3,3]
    rot_matrix = quaternion_to_matrix(quats)
    # import pdb; pdb.set_trace()
    wrist_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
    grab_pos = wrist_keypts[grab_idx].clone()
    grab_pos[0] += WRIST_TO_COLLISION
    ref_dists = torch.norm(wrist_keypts[1:] - wrist_keypts[:-1], dim=1)

    # Optimization loop
    for i in range(3000):
        optimizer.zero_grad()
        if i == 399 :
            debug = True
        else:
            debug = False
        cost = compute_cost(joint_angles, trans, quats, debug=debug)
        t1 = time.time()
        cost.backward()
        t2 = time.time()
        optimizer.step()
        if cost.item() < 0.016:
            break
        if i % 100 == 0:
            print(f"Step {i}, Cost: {cost.item()}")

    global_pose, joints = motion_data['global_pose'], motion_data['joints']
    joints_new_ = joints.clone()  # Copy the original joints
    joints_new_[:,active_joint_ids] = joint_angles.data
    num_timesteps = joints.shape[0]
    joints_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, joints_new_.shape[1]), dtype=joints_new_.dtype)
    trans_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, 3), dtype=joints_new_.dtype)
    quats_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, 4), dtype=joints_new_.dtype)
    trans_new[INTERP_AMT + PAUSE_AMT:, :] = target_trans.clone()
    quats_new[INTERP_AMT + PAUSE_AMT:, :] = target_quats.clone()
    joints_new[INTERP_AMT + PAUSE_AMT:, :] = joints_new_.clone()
    quats_new[:INTERP_AMT + PAUSE_AMT, 0] = 1.0  # Set the first quaternion component to 1.0


    joints_new[:PAUSE_AMT, :] = torch.tensor(init_joint_angles).unsqueeze(0).repeat(PAUSE_AMT, 1)
    joints_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = joints_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
        (joints_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - joints_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
        torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)

    # Make left arm non functional
    joints_new[:,13:20] = 0.
    joints_new[:, 13] = 0.35
    joints_new[:, 14] = 0.16
    joints_new[:, 16] = 0.87


    first_viol_i = 0
    q_dict = {name: joints_new[:, i] for i, name in enumerate( joint_names ) }
    rot_matrix = quaternion_to_matrix(quats_new)
    fk_results = chain.forward_kinematics(q_dict)
    pos_right_ankle = fk_results["right_ankle_roll_link"].get_matrix()[:,:3,3]  # shape (N, 3)
    pos_right_ankle = torch.bmm(pos_right_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_new
    pos_left_ankle = fk_results["left_ankle_roll_link"].get_matrix()[:,:3,3]  # shape (N, 3)
    pos_left_ankle = torch.bmm(pos_left_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_new
    # import pdb; pdb.set_trace()

    if pos_right_ankle[PAUSE_AMT+INTERP_AMT,0] < pos_left_ankle[PAUSE_AMT+INTERP_AMT,0]:
        trans_new[:PAUSE_AMT, 2] -= pos_right_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = pos_right_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_right_ankle[:PAUSE_AMT, :2]
        trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
            (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
            torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)
    else :
        trans_new[:PAUSE_AMT, 2] -= pos_left_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = pos_left_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_left_ankle[:PAUSE_AMT, :2]
        trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
            (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
            torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)

    # Interpolate quats from PAUSE_AMT to PAUSE_AMT + INTERP_AMT

    quats_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = slerp(
        quats_new[PAUSE_AMT-1:PAUSE_AMT, :],
        quats_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :],
        torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)
    )

    Ts_world_root = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(quats_new)),jnp.array(trans_new)) 

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    with open(SAVE_DIR + pkl_path[:-4] + "_n.pkl","wb") as f:
        pickle.dump({"global_pose": Ts_world_root , "joints": joints_new, "global_position": trans_new, "grab_pos": motion_data["grab_pos"], "grab_idx": motion_data["grab_idx"]+PAUSE_AMT+INTERP_AMT}, f)

if VISUALIZE:
    # Define cuboid dimensions (width, height, depth)
    width, height, depth = 1., OFFSET_Z, 2.0

    # Create a cuboid using Trimesh
    cuboid = trimesh.creation.box(extents=(width, depth, height))

    # Optionally, apply a transformation (e.g., move it to x=1, y=0.5, z=0)
    transform = np.eye(4)
    transform[:3, 3] = [grab_pos[0]+0.5+OFFSET_X, 0., OFFSET_Z/2.]
    cuboid.apply_transform(transform)

    # Add the cuboid to Viser
    server.scene.add_mesh_trimesh(
        name="my_cuboid",
        mesh=cuboid,
    )

    print("Started?")
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                time.sleep(0.1)
            tstep = timestep_slider.value
            base_frame_new.wxyz = np.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame_new.position = np.array(Ts_world_root.wxyz_xyz[tstep][4:]) + np.array([0, 0, 0.035])  # Adjust for the height of the robot's base
            urdf_vis_new.update_cfg(np.array(joints_new[tstep]))
