import os
from .paths import BACKEND_ROOT, add_external_paths
from .reference_plan import Generation

add_external_paths()
generation = Generation("stand", __file__)
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
import tqdm
import torch.optim as optim
from isaac_utils.rotations import quaternion_to_matrix, slerp

FIRST_FRAME = 50

# Load URDF and create kinematic chain
urdf_path = str(
    BACKEND_ROOT / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
)
chain = pk.build_chain_from_urdf(open(urdf_path).read())
pkl_path = generation.plan["base"][generation.kind]
VIS = False
SAVE_FOLDER = "Open_Drawer_sim2"
X_MIN = 0.28
X_MAX = 0.3
Y_MIN = -0.2
Y_MAX = -0.1
Z_MIN = 0.7
Z_MAX = 0.8

NX = 2
NY = 10
NZ = 10
# Generate normalized grid
x = np.linspace(X_MIN, X_MAX, NX)
y = np.linspace(Y_MIN, Y_MAX, NY)
z = np.linspace(Z_MIN, Z_MAX, NZ)
SHIFT_Xs, SHIFT_Ys, SHIFT_Zs = np.meshgrid(x, y, z)
SHIFT_Xs = SHIFT_Xs.flatten()
SHIFT_Ys = SHIFT_Ys.flatten()
SHIFT_Zs = SHIFT_Zs.flatten()
if generation.enabled:
    SHIFT_Xs, SHIFT_Ys, SHIFT_Zs = generation.points.T
INTERP_AMT = 10
PAUSE_AMT = 10


def quad_func(x):
    """Quadratic function for trajectory generation."""
    return 1 - (x - 1) ** 2


def generate_ref_traj(
    wrist_pos, target_x=0.45, target_z=0.8, target_y=-0.1, door_open_amt=0.25
):
    global transformed_keypts_ref
    transformed_keypts_ref = np.zeros((np.shape(wrist_pos)[0], 3))
    wrist_pos[0, 0] = 0.0
    wrist_pos[0, 1] = -0.25
    wrist_pos[0, 2] = 0.7
    i = np.arange(40)
    xs = wrist_pos[0, 0] + (target_x - wrist_pos[0, 0]) * (i / 39.0)
    ys = wrist_pos[0, 1] + (target_y - wrist_pos[0, 1]) * (i / 39.0)
    zs = wrist_pos[0, 2] + (target_z - wrist_pos[0, 2]) * quad_func(i / 39.0)
    transformed_keypts_ref[:40, 0] = xs
    transformed_keypts_ref[:40, 1] = ys
    transformed_keypts_ref[:40, 2] = zs

    # Freeze for 1.5s
    transformed_keypts_ref[40:70, 0] = xs[-1:]
    transformed_keypts_ref[40:70, 1] = ys[-1:]
    transformed_keypts_ref[40:70, 2] = zs[-1:]

    # Open the drawer
    i = np.arange(40)
    target_open_pos = [target_x - door_open_amt, target_y, target_z]
    xs = target_x + (target_open_pos[0] - target_x) * (i / 39.0)
    ys = target_y + (target_open_pos[1] - target_y) * (i / 39.0)
    zs = target_z + (target_open_pos[2] - target_z) * (i / 39.0)
    transformed_keypts_ref[70:110, 0] = xs
    transformed_keypts_ref[70:110, 1] = ys
    transformed_keypts_ref[70:110, 2] = zs

    transformed_keypts_ref[110:, 0] = xs[-1:]
    transformed_keypts_ref[110:, 1] = ys[-1:]
    transformed_keypts_ref[110:, 2] = zs[-1:]

    transformed_keypts_ref = torch.tensor(transformed_keypts_ref, dtype=torch.float32)


def compute_cost(
    joint_angles, trans, quats, offset_x=-0.55, offset_z=-0.0, debug=False
):
    # L2 cost to target
    l2_cost = 0.0 * torch.nn.functional.mse_loss(
        joint_angles, target_joint_angles[:, active_joint_ids]
    )

    # Forward kinematics to get link positions
    q_dict = {name: joint_angles[:, i] for i, name in enumerate(active_joint_names)}
    for id in inactive_joint_ids:
        name = joint_names[id]
        q_dict[name] = target_joint_angles[:, id]
    fk_results = chain.forward_kinematics(q_dict)
    # Collect all link positions
    cost2 = torch.zeros(joint_angles.shape[0])
    rot_matrix = quaternion_to_matrix(quats)
    i = 0
    for link_name, tf in fk_results.items():
        if link_name == "right_wrist_yaw_link":
            pos = tf.get_matrix()[:, :3, 3]  # shape (N, 3)
            cost2 += 0.1 * torch.mean(torch.norm(pos[1:] - pos[:-1], dim=1))
            transformed_keypts = (
                torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans
            )
            if debug:
                print(transformed_keypts[40], transformed_keypts_ref[40])

            diff = transformed_keypts - transformed_keypts_ref
            cost2 += 10 * (
                torch.abs(diff[:, 0]) + torch.abs(diff[:, 1]) + torch.abs(diff[:, 2])
            )  # L2 cost for position
            rot_mat = tf.get_matrix()[:, :3, :3]  # shape (N, 3, 3)
            rot_mat = torch.bmm(rot_matrix, rot_mat)  # Adjust for the coordinate system
            rot_mat_ref = torch.tensor(
                [
                    [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 0, -1],
                    [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                ],
                dtype=torch.float32,
            ).T.to(rot_mat.device)  # Adjust for the coordinate system

            ja_diff = 0.03 * torch.sum(
                torch.abs(joint_angles[1:] - joint_angles[:-1]), dim=1
            )
            cost2[:-1] += ja_diff
            cost2[:] += (
                10.0 * (joint_angles[:, 21] + 0.15) * (joint_angles[:, 21] > -0.15)
            )
            # Multiply rot_mat with rot_mat_ref
            rot_mat = torch.bmm(
                rot_mat, rot_mat_ref.unsqueeze(0).expand(rot_mat.shape[0], -1, -1)
            )
            # Convert rotation matrix to quaternion
            angle = torch.acos(
                torch.clamp(
                    (rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2] - 1) / 2,
                    -0.999,
                    0.999,
                )
            )
            cost2 += 0.3 * angle  # L2 cost for orientation
        else:
            i += 1
            continue
        i += 1

    return l2_cost + torch.mean(cost2)


target_positions = []
joint_angles_next = []

urdf = yourdfpy.URDF.load(
    str(
        BACKEND_ROOT
        / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    )
)

heightmap = np.zeros(
    (1000, 1000), dtype=np.float32
)  # Dummy heightmap for visualization

heightmap = pk2.collision.Heightmap(
    pose=jaxlie.SE3.identity(),
    size=jnp.array([0.01, 0.01, 1.0]),
    height_data=heightmap,
)

if not generation.enabled or VIS:
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", False)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())
    timestep_slider = server.gui.add_slider("timestep", 0, 196, 1, 0)


for i in tqdm.tqdm(range(len(SHIFT_Xs))):
    SHIFT_X = SHIFT_Xs[i]
    SHIFT_Y = SHIFT_Ys[i]
    SHIFT_Z = SHIFT_Zs[i]
    point_index = i
    if generation.enabled:
        resumed = generation.resume(point_index)
        if resumed is not None:
            target_positions.append([SHIFT_X, SHIFT_Y, SHIFT_Z])
            joint_angles_next.append(resumed)
            continue

    motion_data = pkl.load(open(pkl_path, "rb"))
    target_trans = torch.tensor(np.array(motion_data["global_position"])).clone()
    target_quats = torch.tensor(
        np.array(motion_data["global_pose"].rotation().wxyz)
    ).clone()

    # Target joint angles (example, replace with your actual target)
    target_joint_angles = torch.tensor(np.array(motion_data["joints"])).clone()

    joint_names = [
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
        "right_wrist_yaw_joint",
    ]
    inactive_joint_names = []  # , 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    active_joint_ids = [
        i for i, name in enumerate(joint_names) if name not in inactive_joint_names
    ]
    active_joint_names = [
        name for name in joint_names if name not in inactive_joint_names
    ]
    inactive_joint_ids = [
        i for i, name in enumerate(joint_names) if name in inactive_joint_names
    ]
    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate(joint_names)}
    fk_results_ref = chain.forward_kinematics(q_dict)
    # copy a tensor
    # Initial guess for joint angles (can be zeros or random)
    target_joint_angles[1:] = target_joint_angles[0:1]  # Freeze all but the first frame
    joint_angles = torch.nn.Parameter(
        torch.tensor(target_joint_angles[:, active_joint_ids]).clone()
    )  # Only optimize active joints
    target_trans *= 0.0

    target_quats *= 0.0
    target_quats[:, 0] = 1.0

    joint_names = [
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
        "right_wrist_yaw_joint",
    ]
    init_joint_angles = [
        -0.2,
        0.0,
        0.0,
        0.42,
        -0.23,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.42,
        -0.23,
        0.0,
        0.0,
        0.35,
        0.16,
        0.0,
        0.87,
        0.0,
        0.0,
        0.0,
        0.35,
        -0.16,
        0.0,
        0.87,
        0.0,
        0.0,
        0.0,
    ]

    target_joint_angles[:, :12] = torch.tensor(init_joint_angles).unsqueeze(0)[:, :12]

    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate(joint_names)}
    rot_matrix = quaternion_to_matrix(target_quats)
    fk_results = chain.forward_kinematics(q_dict)
    pos_right_ankle = fk_results["right_ankle_roll_link"].get_matrix()[:, :3, 3]
    pos_right_ankle = (
        torch.bmm(pos_right_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0]
        + target_trans
    )
    target_trans[:, 2] -= pos_right_ankle[:, 2]

    trans = torch.tensor(target_trans)  # Translation offset
    quats = torch.tensor(target_quats)  # Quaternion offset

    if i == 0:
        joint_angles = torch.nn.Parameter(
            torch.tensor(target_joint_angles[:, active_joint_ids]).clone()
        )
    else:
        current_position = np.array([[SHIFT_X, SHIFT_Y, SHIFT_Z]])
        dists = np.linalg.norm(np.array(target_positions) - current_position, axis=1)
        min_i = np.argmin(dists)
        joint_angles = torch.nn.Parameter(
            torch.tensor(joint_angles_next[min_i]).clone()
        )

    target_positions.append([SHIFT_X, SHIFT_Y, SHIFT_Z])
    optimizer = optim.Adam([joint_angles], lr=0.01)

    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate(joint_names)}
    fk_results = chain.forward_kinematics(q_dict)
    tf = fk_results["right_wrist_yaw_link"]
    pos = tf.get_matrix()[:, :3, 3]  # shape (N, 3)

    rot_matrix = quaternion_to_matrix(quats)
    wrist_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans

    generate_ref_traj(
        wrist_keypts.detach().numpy(),
        target_x=SHIFT_X,
        target_z=SHIFT_Z,
        target_y=SHIFT_Y,
        door_open_amt=0.2,
    )

    n_steps = 2000
    if i > 0:
        n_steps = 500

    # Optimization loop
    for i in range(n_steps):
        optimizer.zero_grad()
        if i == 499:
            debug = True
        else:
            debug = False
        cost = compute_cost(joint_angles, trans, quats, debug=debug)
        t1 = time.time()
        cost.backward()
        t2 = time.time()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step {i}, Cost: {cost.item()}")

    joint_angles_next.append(joint_angles.data.cpu().numpy())
    joints_new_ = target_joint_angles.clone()
    num_timesteps = joints_new_.shape[0]
    joints_new_[:, active_joint_ids] = joint_angles.data
    num_timesteps = target_joint_angles.shape[0]

    joints_new = torch.zeros(
        (joints_new_.shape[0] + INTERP_AMT + PAUSE_AMT, joints_new_.shape[1]),
        dtype=joints_new_.dtype,
    )
    trans_new = torch.zeros(
        (joints_new_.shape[0] + INTERP_AMT + PAUSE_AMT, 3), dtype=joints_new_.dtype
    )
    quats_new = torch.zeros(
        (joints_new_.shape[0] + INTERP_AMT + PAUSE_AMT, 4), dtype=joints_new_.dtype
    )
    trans_new[INTERP_AMT + PAUSE_AMT :, :] = target_trans.clone()
    quats_new[INTERP_AMT + PAUSE_AMT :, :] = target_quats.clone()
    joints_new[INTERP_AMT + PAUSE_AMT :, :] = joints_new_.clone()
    quats_new[: INTERP_AMT + PAUSE_AMT, 0] = 1.0

    joints_new[:PAUSE_AMT, :] = (
        torch.tensor(init_joint_angles).unsqueeze(0).repeat(PAUSE_AMT, 1)
    )
    joints_new[PAUSE_AMT : PAUSE_AMT + INTERP_AMT, :] = joints_new[
        PAUSE_AMT - 1 : PAUSE_AMT, :
    ] + (
        joints_new[PAUSE_AMT + INTERP_AMT : PAUSE_AMT + INTERP_AMT + 1, :]
        - joints_new[PAUSE_AMT - 1 : PAUSE_AMT, :]
    ) * torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)

    joints_new[:, 13:20] = 0.0
    joints_new[:, 13] = 0.35
    joints_new[:, 14] = 0.16
    joints_new[:, 16] = 0.87

    first_viol_i = 0
    q_dict = {name: joints_new[:, i] for i, name in enumerate(joint_names)}
    rot_matrix = quaternion_to_matrix(quats_new)
    fk_results = chain.forward_kinematics(q_dict)
    pos_right_ankle = fk_results["right_ankle_roll_link"].get_matrix()[:, :3, 3]
    pos_right_ankle = (
        torch.bmm(pos_right_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0]
        + trans_new
    )
    pos_left_ankle = fk_results["left_ankle_roll_link"].get_matrix()[:, :3, 3]
    pos_left_ankle = (
        torch.bmm(pos_left_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0]
        + trans_new
    )

    if (
        pos_right_ankle[PAUSE_AMT + INTERP_AMT, 0]
        < pos_left_ankle[PAUSE_AMT + INTERP_AMT, 0]
    ):
        trans_new[:PAUSE_AMT, 2] -= pos_right_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = (
            pos_right_ankle[PAUSE_AMT + INTERP_AMT : PAUSE_AMT + INTERP_AMT + 1, :2]
            - pos_right_ankle[:PAUSE_AMT, :2]
        )
        trans_new[PAUSE_AMT : PAUSE_AMT + INTERP_AMT, :] = trans_new[
            PAUSE_AMT - 1 : PAUSE_AMT, :
        ] + (
            trans_new[PAUSE_AMT + INTERP_AMT : PAUSE_AMT + INTERP_AMT + 1, :]
            - trans_new[PAUSE_AMT - 1 : PAUSE_AMT, :]
        ) * torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)
    else:
        trans_new[:PAUSE_AMT, 2] -= pos_left_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = (
            pos_left_ankle[PAUSE_AMT + INTERP_AMT : PAUSE_AMT + INTERP_AMT + 1, :2]
            - pos_left_ankle[:PAUSE_AMT, :2]
        )
        trans_new[PAUSE_AMT : PAUSE_AMT + INTERP_AMT, :] = trans_new[
            PAUSE_AMT - 1 : PAUSE_AMT, :
        ] + (
            trans_new[PAUSE_AMT + INTERP_AMT : PAUSE_AMT + INTERP_AMT + 1, :]
            - trans_new[PAUSE_AMT - 1 : PAUSE_AMT, :]
        ) * torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)

    quats_new[PAUSE_AMT : PAUSE_AMT + INTERP_AMT, :] = slerp(
        quats_new[PAUSE_AMT - 1 : PAUSE_AMT, :],
        quats_new[PAUSE_AMT + INTERP_AMT : PAUSE_AMT + INTERP_AMT + 1, :],
        torch.linspace(0, 1, INTERP_AMT).unsqueeze(1),
    )

    Ts_world_root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.array(quats_new.cpu())), jnp.array(trans_new.cpu())
    )

    if generation.enabled:
        generation.save(
            point_index,
            {
                "global_pose": Ts_world_root,
                "joints": joints_new,
                "global_position": trans,
                "open_drawer_pos": motion_data["open_drawer_pos"],
                "grab_idx": 40 + PAUSE_AMT + INTERP_AMT,
            },
            joint_angles_next[-1],
            cost.item(),
        )
        continue
    os.makedirs("../" + SAVE_FOLDER, exist_ok=True)
    # Get filename from pkl_path
    filename = (
        os.path.basename(pkl_path)[:-4]
        + "_"
        + str(int(SHIFT_X * 1000))
        + "_"
        + str(int(SHIFT_Y * 1000))
        + "_"
        + str(int(SHIFT_Z * 1000))
        + ".pkl"
    )
    with open("../" + SAVE_FOLDER + "/" + filename, "wb") as f:
        pickle.dump(
            {
                "global_pose": Ts_world_root,
                "joints": joints_new,
                "global_position": trans,
                "open_drawer_pos": motion_data["open_drawer_pos"],
                "grab_idx": 40 + PAUSE_AMT + INTERP_AMT,
            },
            f,
        )


if VIS:
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                time.sleep(0.1)
            tstep = timestep_slider.value
            base_frame.wxyz = np.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = np.array(
                Ts_world_root.wxyz_xyz[tstep][4:]
            ) + np.array([0, 0, 0.035])  # Adjust for the height of the robot's base
            urdf_vis.update_cfg(np.array(joints_new[tstep]))
