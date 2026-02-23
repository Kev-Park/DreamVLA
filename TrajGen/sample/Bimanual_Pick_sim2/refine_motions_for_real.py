import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import isaaclab.utils.math as math_utils
import torch.optim as optim
from isaac_utils.rotations import(
    quat_conjugate,
    quaternion_to_matrix,
    slerp
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def compute_cost(joint_angles, trans, quats):
    # L2 cost to target
    l2_cost = 0.*torch.nn.functional.mse_loss(joint_angles, target_joint_angles[:, active_joint_ids])
    q_dict = {name: joint_angles[:, i] for i, name in enumerate( active_joint_names ) }
    for id in inactive_joint_ids:
        name = joint_names[id]
        q_dict[name] = target_joint_angles[:, id]
    fk_results = chain.forward_kinematics(q_dict)
    cost2 = torch.zeros(joint_angles.shape[0], device=DEVICE)
    rot_matrix = quaternion_to_matrix(quats)
    i = 0
    ja_diff = 0.03*torch.sum(torch.abs(joint_angles[1:] - joint_angles[:-1]), dim=1)
    cost2[:-1] += ja_diff

    for link_name, tf in fk_results.items():
        if link_name == "left_ankle_roll_link":
            pos = tf.get_matrix()[:,:3,3]
            cost2 += 0.*torch.mean(torch.norm(pos[1:] - pos[:-1], dim=1))
            transformed_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
            ref_pos = torch.tensor([[feet_x, feet_gap, 0.]], device=DEVICE)
            
            # print(transformed_keypts[30:50])
            vals = transformed_keypts - ref_pos
            cost2 += torch.norm(vals, dim=1)

        if link_name == "right_ankle_roll_link":
            pos = tf.get_matrix()[:,:3,3]
            cost2 += 0.*torch.mean(torch.norm(pos[1:] - pos[:-1], dim=1))
            transformed_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans
            ref_pos = torch.tensor([[feet_x, -feet_gap, 0.]], device=DEVICE)
            
            # import pdb; pdb.set_trace()
            vals = transformed_keypts - ref_pos
            cost2 += torch.norm(vals, dim=1)
        
        
        if link_name == "right_rubber_hand" or link_name == "left_rubber_hand":
            rot_mat = tf.get_matrix()[:,:3,:3]
            rot_mat = torch.bmm(rot_matrix, rot_mat)
            rot_mat_ref = torch.tensor([[1, 0, 0], 
                                        [0, 1, 0], 
                                        [0, 0, 1]], dtype=torch.float32, device=DEVICE).T
            rot_mat = torch.bmm(rot_mat, rot_mat_ref.unsqueeze(0).expand(rot_mat.shape[0], -1, -1))
            angle = torch.acos(torch.clamp((rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2] - 1) / 2, -0.999, .999))
            # import pdb; pdb.set_trace()
            cost2 += 0.3*angle
        
        i += 1
    return l2_cost + torch.mean(cost2)

# Load URDF and create kinematic chain
urdf_path = '../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
chain = pk.build_chain_from_urdf(open(urdf_path).read())
pkl_path = '0_n_500.pkl'
SMOOTH_AMT = 20
PAUSE_AMT = 10
INTERP_AMT = 10
SAVE_DIR = "../Bimanual_Pick_real/"

motion_data = pkl.load(open(pkl_path, 'rb'))

feet_gap = 0.15
feet_x = 0.065

SHIFT_FACTOR = 1.

target_trans = torch.tensor(np.array(motion_data['global_position'])).clone()[:200]
target_quats = torch.tensor(np.array(motion_data['global_pose'].rotation().wxyz)).clone()[:200]
target_quats_roll, target_quats_pitch, target_quats_yaw = math_utils.euler_xyz_from_quat(target_quats)
target_quats_roll[:] = 0.
target_quats_yaw[:] = 0.
target_quats = math_utils.quat_from_euler_xyz(target_quats_roll, target_quats_pitch*SHIFT_FACTOR, target_quats_yaw)
target_trans[:, 2] = target_trans[0, 2] + SHIFT_FACTOR * (target_trans[:, 2] - target_trans[0, 2])

# Target joint angles (example, replace with your actual target)
target_joint_angles = torch.tensor(np.array(motion_data['joints'])).clone()[:200]
# print(motion_data.keys())
grab_pos = torch.tensor(np.array(motion_data['squat_pos_real'])).clone()



joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
init_joint_angles = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0., 0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.]
inactive_joint_names = ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
active_joint_ids = [i for i, name in enumerate(joint_names) if name not in inactive_joint_names]
active_joint_names = [name for name in joint_names if name not in inactive_joint_names]
inactive_joint_ids = [i for i, name in enumerate(joint_names) if name in inactive_joint_names]

joint_angles = torch.nn.Parameter(torch.tensor(target_joint_angles[:, active_joint_ids], device=DEVICE).clone())
trans = torch.tensor(target_trans, device=DEVICE)
quats = torch.tensor(target_quats, device=DEVICE)
optimizer = optim.Adam([joint_angles], lr=0.003)

for j in range(0):
    # print(j)
    optimizer.zero_grad()
    if j == 399 :
        debug = True
    else:
        debug = False
    t1 = time.time()
    cost = compute_cost(joint_angles, trans, quats)
    t2 = time.time()
    
    t1 = time.time()
    cost.backward()
    t2 = time.time()
    
    # if cost.item() < 0.0067 :
    #     break
    
    optimizer.step()
    if j % 100 == 0:
        print(f"Step {j}, Cost: {cost.item()}")

heightmap = np.zeros((1000, 1000), dtype=np.float32)  # Dummy heightmap for visualization
joints = motion_data['joints'][:200]
joints_new_ = joints.clone()  # Copy the original joints
num_timesteps = joints.shape[0]
joints_new_[20:,active_joint_ids] = joint_angles.data[20:]
joints_new = joints_new_.clone()   
trans_new = target_trans.clone()
quats_new = target_quats.clone()


heightmap = pk2.collision.Heightmap(
    pose=jaxlie.SE3.identity(),
    size=jnp.array([0.01, 0.01, 1.0]),
    height_data=heightmap,
)


# asset_dir = Path(__file__).parent / "retarget_helpers" / "humanoid" / "amass"
urdf = yourdfpy.URDF.load('../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')

server = viser.ViserServer(port=8081)
base_frame = server.scene.add_frame("/base", show_axes=False)
base_frame_new = server.scene.add_frame("/base_new", show_axes=False)

urdf_vis_new = ViserUrdf(server, urdf, root_node_name="/base_new")
playing = server.gui.add_checkbox("playing", False)
timestep_slider = server.gui.add_slider("timestep", 0, 2*(num_timesteps - 1) , 1, 0)
server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

Ts_world_root = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(quats_new)),jnp.array(trans_new))

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with open(SAVE_DIR + pkl_path[:-4] + "_n.pkl","wb") as f:
    pickle.dump({"global_pose": Ts_world_root , "joints": joints_new, "global_position": trans_new, "squat_pos_real": motion_data["squat_pos_real"], "grab_idx": motion_data["grab_idx"]+PAUSE_AMT+INTERP_AMT}, f)


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
