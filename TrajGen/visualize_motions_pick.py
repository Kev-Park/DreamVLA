import time
from typing import Tuple, TypedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
# import jaxls
import numpy as onp
import pyroki as pk
import pytorch_kinematics as pk2
import trimesh
import viser
from viser.extras import ViserUrdf
# from pyroki.collision import colldist_from_sdf, collide
# from robot_descriptions.loaders.yourdfpy import load_robot_description

# from retarget_helpers._utils import (
#     SMPL_JOINT_NAMES,
#     create_conn_tree,
#     get_humanoid_retarget_indices_g1_27dof,
#     get_humanoid_retarget_indices,
# )
import pickle
import yourdfpy
import torch
from isaac_utils.rotations import(
    quat_conjugate,
    quaternion_to_matrix
)
import pyroki as pk
import numpy as np

class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""
    floor_contact: float
    """Floor contact weight, to place the robot's foot on the floor."""
    root_smoothness: float
    """Root smoothness weight, to penalize the robot's root from jittering too much."""
    foot_skating: float
    """Foot skating weight, to penalize the robot's foot from moving when it is in contact with the floor."""
    world_collision: float
    """World collision weight, to penalize the robot from colliding with the world."""

def load_pickle(path):
    
    with open(path, "rb") as f:
        DATA = pickle.load(f)
    
    return DATA

def get_keypts(joint_angles, joint_names, pk2_robot):
    
        q_dict = {name: joint_angles[:, i] for i, name in enumerate( joint_names ) }

        tf_dict = pk2_robot.forward_kinematics( q_dict )

        keypts = torch.zeros((len(joint_angles), len(tf_dict), 3))
        # print(len(tf_dict))
        # exit(0)
        cntr = 0 
        
        for name in tf_dict.keys():
            print("1, #", name)
            tf_val = tf_dict[name].get_matrix()  
            t = tf_val[:, :3, -1 ]
            # print(name, t)
            keypts[:, cntr , :] = t 
            cntr += 1 
        # print(keypts)
        # exit(0)
        return keypts


def transform_keypts(keypts, quat, translation):
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



def _cylinder_between(p0: onp.ndarray, p1: onp.ndarray, radius: float,
                      color=(255, 165, 0, 200), sections: int = 20) -> "trimesh.Trimesh | None":
    """Return a trimesh.Trimesh cylinder aligned from p0 to p1."""
    direction = onp.array(p1, dtype=float) - onp.array(p0, dtype=float)
    length = onp.linalg.norm(direction)
    if length < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    # Rotate default Z-axis cylinder to align with 'direction'
    direction /= length
    z = onp.array([0.0, 0.0, 1.0])
    v = onp.cross(z, direction)
    s = onp.linalg.norm(v)
    c = onp.dot(z, direction)
    if s < 1e-6:
        rot = onp.eye(3) if c > 0 else onp.diag([1.0, -1.0, -1.0])
    else:
        vx = onp.array([[ 0,    -v[2],  v[1]],
                        [ v[2],  0,    -v[0]],
                        [-v[1],  v[0],  0   ]])
        rot = onp.eye(3) + vx + vx @ vx * (1.0 - c) / (s ** 2)
    tf = onp.eye(4)
    tf[:3, :3] = rot
    tf[:3,  3] = (onp.array(p0, dtype=float) + onp.array(p1, dtype=float)) / 2.0
    cyl.apply_transform(tf)
    cyl.visual.face_colors = list(color)
    return cyl


def main(show_segments=False, forearm_length=0.25, forearm_thickness=0.05,
         hand_length=0.15, hand_thickness=0.04, collision_cylinder_radius=0.05):
    
    urdf = yourdfpy.URDF.load('../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    #urdf = yourdfpy.URDF.load('../../HumanoidVerse/humanoidverse/data/robots/g1/g1_paddle_hand_rigid.urdf')
    robot = pk.Robot.from_urdf(urdf)
    # robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    print( len(robot.joints.actuated_names ) )

    print("1. Loading data...")
    HUMAN_DATA = load_pickle(human_data_path)
    G1_DATA = load_pickle(g1_data_path)
    #print(G1_DATA["grab_idx"])
    #print(G1_DATA)
    # exit(0)
    smpl_keypoints = HUMAN_DATA['poses'][0, :]
    #heightmap = HUMAN_DATA['height_map'].numpy()
    heightmap = onp.zeros((1000, 1000), dtype=onp.float32)  # Dummy heightmap for visualization

    if "body_pos_w" in G1_DATA:
        # Refined format (Pick_sim2)
        body_pos_w = np.array(G1_DATA["body_pos_w"])
        body_quat_w = np.array(G1_DATA["body_quat_w"])
        global_pose = np.concatenate([body_pos_w[:, 0, :], body_quat_w[:, 0, :]], axis=-1)
        joints = np.array(G1_DATA["joint_pos"])
    else:
        # Retargeted format (Pick_sim1)
        global_position = np.array(G1_DATA["global_position"])
        global_orientation = np.array(G1_DATA["global_pose"].rotation().wxyz)
        global_pose = np.concatenate([global_position, global_orientation], axis=-1)
        joints = np.array(G1_DATA["joints"])

    grab_pos = onp.array(G1_DATA.get("grab_pos", onp.zeros(3)))
    grab_idx = G1_DATA.get("grab_idx", None)
    hand_tip_traj = onp.array(G1_DATA["hand_tip_traj"]) if "hand_tip_traj" in G1_DATA else None

    # Debug overlay fields (only present in Pick_sim2 pkls produced by refine_motions_al.py)
    debug_capsule_obs_pos  = onp.array(G1_DATA["debug_capsule_obs_pos"])  if "debug_capsule_obs_pos"  in G1_DATA else None
    debug_wrist_world_traj = onp.array(G1_DATA["debug_wrist_world_traj"]) if "debug_wrist_world_traj" in G1_DATA else None
    debug_hand_world_traj  = onp.array(G1_DATA["debug_hand_world_traj"])  if "debug_hand_world_traj"  in G1_DATA else None
    if debug_capsule_obs_pos is not None:
        print(f"[DEBUG] optimizer capsule_obs_pos  = {debug_capsule_obs_pos.round(3)}")
        print(f"[DEBUG] viewer    grab_pos (red sph) = {grab_pos.round(3)}")
        print(f"[DEBUG] XY offset (capsule - grab_pos): {(debug_capsule_obs_pos[:2] - grab_pos[:2]).round(3)} "
              f"(should both sit at the cylinder axis)")

    num_timesteps = joints.shape[0]

    heightmap = pk.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.01, 1.0]),
        height_data=heightmap,
    )

    # asset_dir = Path(__file__).parent / "retarget_helpers" / "humanoid" / "amass"

    #CREATE VISUALIZER 
    server = viser.ViserServer(host="0.0.0.0", port=7860, show=False, verbose=True)

    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

    # weights = pk.viewer.WeightTuner(
    #     server,
    #     RetargetingWeights(
    #         local_alignment=2.0,
    #         global_alignment=1.0,
    #         floor_contact=1.0,
    #         root_smoothness=1.0,
    #         foot_skating=1.0,
    #         world_collision=1.0,
    #     ),  # type: ignore
    # )
    # import pdb; pdb.set_trace()
    # global_pose[:,2] += 0.035

    """CHANGED FORMAT FOR COMMAND INPUT TESTING"""
    #Ts_world_root, joints = global_pose , joints 
    #positions = onp.array(Ts_world_root.translation())
    #orientations = onp.array(Ts_world_root.rotation().wxyz)

    Ts_world_root, joints = global_pose , joints 
    Ts_world_root = onp.array(Ts_world_root)
    positions = Ts_world_root[:, :3] 
    orientations = Ts_world_root[:, 3:7] #first 4 or second?
    #orientations = orientations[:, [3, 0, 1, 2]] 

    # Ts_world_root[:30] = Ts_world_root[30]
    # positions[:65] = positions[65:66] + (positions[:65]-positions[65:66])*0.6
    # orientations[:30] = orientations[30]
    # print(len(joints[0]))

    """CHANGED FORMAT FOR COMMAND INPUT TESTING"""
    joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    urdf_path = '../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())

    #joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint','waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    #urdf_path = '../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_paddle_hand_rigid.urdf'
    #pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())
    
    keypts = get_keypts(torch.tensor(joints), joint_names , pk2_robot=pk2_robot) 
    global_keypts = transform_keypts(torch.tensor(keypts), torch.tensor(orientations), torch.tensor(positions + onp.array([0, 0, 0.035]))).numpy()
    
    # Table cuboid matching refine_motions.py cost geometry:
    #   x_edge = wrist x at grab_idx (link 36, WRIST_TO_COLLISION and OFFSET_X cancel)
    #   z_table = OFFSET_Z = 0.86 (fixed absolute height)
    OFFSET_Z = 0.86
    _gi = grab_idx if grab_idx is not None else -1
    x_edge = global_keypts[_gi, 36, 0]
    table_w, table_h, table_d = 1.0, OFFSET_Z, 2.0  # matches refine_motions VISUALIZE block
    cuboid = trimesh.creation.box(extents=(table_w, table_d, table_h))
    transform = onp.eye(4)
    transform[:3, 3] = [x_edge + table_w / 2., 0., table_h / 2.]
    cuboid.apply_transform(transform)
    server.scene.add_mesh_trimesh(name="my_cuboid", mesh=cuboid)

    # Collision cylinder — vertical infinite cylinder at grab XY site (matches capsule_collision_cost)
    if show_segments and onp.any(grab_pos != 0):
        cyl = trimesh.creation.cylinder(radius=collision_cylinder_radius, height=3.0, sections=32)
        cyl_tf = onp.eye(4)
        cyl_tf[:3, 3] = [grab_pos[0], grab_pos[1], 1.5]
        cyl.apply_transform(cyl_tf)
        cyl.visual.face_colors = [255, 80, 80, 80]  # translucent red
        server.scene.add_mesh_trimesh("/collision_cylinder", cyl)

    # Debug: optimizer's obstacle centre — cyan sphere + separate magenta cylinder at its XY
    # If the cyan sphere/cylinder doesn't sit on top of the red one, the obstacle is misplaced.
    if debug_capsule_obs_pos is not None:
        server.scene.add_icosphere(
            "/debug_capsule_obs",
            radius=0.04,
            color=(0, 220, 220),   # cyan
            position=debug_capsule_obs_pos.astype(onp.float64),
        )
        if show_segments:
            dbg_cyl = trimesh.creation.cylinder(radius=collision_cylinder_radius, height=3.0, sections=32)
            dbg_cyl_tf = onp.eye(4)
            dbg_cyl_tf[:3, 3] = [debug_capsule_obs_pos[0], debug_capsule_obs_pos[1], 1.5]
            dbg_cyl.apply_transform(dbg_cyl_tf)
            dbg_cyl.visual.face_colors = [0, 220, 220, 80]  # translucent cyan
            server.scene.add_mesh_trimesh("/debug_collision_cylinder", dbg_cyl)
        print(f"  [DEBUG] Cyan sphere = optimizer capsule_obs_pos.  "
              f"Red sphere = grab_pos from pkl.  They should overlap.")

    # Grab point sphere
    grab_sphere = None
    if onp.any(grab_pos != 0):
        grab_sphere = server.scene.add_icosphere(
            "/grab_point",
            radius=0.04,
            color=(255, 50, 50),
            position=onp.array(grab_pos, dtype=onp.float64),
        )
        print(f"  Grab point at {grab_pos}" + (f", grab frame: {grab_idx}" if grab_idx is not None else ""))
    else:
        print("  No grab_pos found in data — skipping grab point sphere.")


    # (my_cuboid1 removed — superseded by my_cuboid above which matches refine_motions geometry)
    
    print(positions)
    print("Started?")

    # Sphere or segment end-effector visualization
    if not show_segments:
        hand_sphere = server.scene.add_icosphere(
            "/right_hand",
            radius=0.04,
            color=(255, 0, 0),
            position=global_keypts[0, 38, :],
        )
        wrist_sphere = server.scene.add_icosphere(
            "/right_wrist",
            radius=0.03,
            color=(0, 0, 255),
            position=global_keypts[0, 36, :],
        )
        if hand_tip_traj is not None:
            hand_tip_sphere = server.scene.add_icosphere(
                "/hand_tip",
                radius=0.025,
                color=(0, 220, 80),
                position=hand_tip_traj[0],
            )
            print(f"  Hand tip trajectory loaded: {hand_tip_traj.shape[0]} frames")
        else:
            hand_tip_sphere = None
            print("  No hand_tip_traj in pkl — skipping hand tip sphere (Pick_sim1 or old pkl)")

        # Debug spheres — optimizer's view of wrist/hand (yellow = wrist, orange = hand origin)
        # These should perfectly overlap the blue/red spheres above.
        # Any gap = the optimizer is working with wrong positions.
        if debug_wrist_world_traj is not None:
            debug_wrist_sphere = server.scene.add_icosphere(
                "/debug_wrist",
                radius=0.025,
                color=(255, 255, 0),   # yellow
                position=debug_wrist_world_traj[0],
            )
            print("  [DEBUG] Yellow sphere = optimizer wrist pos (should overlap blue sphere)")
        else:
            debug_wrist_sphere = None
        if debug_hand_world_traj is not None:
            debug_hand_sphere = server.scene.add_icosphere(
                "/debug_hand",
                radius=0.025,
                color=(255, 140, 0),   # orange
                position=debug_hand_world_traj[0],
            )
            print("  [DEBUG] Orange sphere = optimizer hand-origin pos (should overlap red sphere)")
        else:
            debug_hand_sphere = None
    else:
        hand_sphere = wrist_sphere = hand_tip_sphere = None
        debug_wrist_sphere = debug_hand_sphere = None
        if hand_tip_traj is not None:
            print(f"  Hand tip trajectory loaded: {hand_tip_traj.shape[0]} frames (segment mode)")
        else:
            print("  No hand_tip_traj in pkl — hand segment will end at hand origin")

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = orientations[tstep]
            base_frame.position = positions[tstep] + onp.array([0, 0, 0.035])  # Adjust for the height of the robot's base
            # base_frame.position[2] += 0.35  # Adjust for the height of the robot's base
            # print(base_frame.position)
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            if not show_segments:
                # Sphere mode — update positions each frame
                hand_sphere.position = global_keypts[tstep, 38, :]       # link 38 = right_rubber_hand
                wrist_sphere.position = global_keypts[tstep, 36, :]      # link 36 = right_wrist_pitch_link
                if hand_tip_sphere is not None:
                    hand_tip_sphere.position = hand_tip_traj[min(tstep, len(hand_tip_traj) - 1)] + onp.array([0, 0, 0.035])
                # Debug: optimizer-view positions (0.035 already baked in — do NOT add again)
                if debug_wrist_sphere is not None:
                    debug_wrist_sphere.position = debug_wrist_world_traj[min(tstep, len(debug_wrist_world_traj) - 1)]
                if debug_hand_sphere is not None:
                    debug_hand_sphere.position = debug_hand_world_traj[min(tstep, len(debug_hand_world_traj) - 1)]
            else:
                # Segment mode — draw forearm and hand as trimesh cylinders
                wrist_pt = global_keypts[tstep, 36, :]
                hand_pt  = global_keypts[tstep, 38, :]
                # Forearm: wrist (36) → hand origin (38)
                forearm_cyl = _cylinder_between(
                    wrist_pt, hand_pt,
                    radius=forearm_thickness,
                    color=(255, 165, 0, 220),
                )
                if forearm_cyl is not None:
                    server.scene.add_mesh_trimesh("/arm_forearm", forearm_cyl)
                # Hand: hand origin (38) → fingertip
                if hand_tip_traj is not None:
                    tip_pt = hand_tip_traj[min(tstep, len(hand_tip_traj) - 1)] + onp.array([0, 0, 0.035])
                else:
                    direction = hand_pt - wrist_pt
                    norm = onp.linalg.norm(direction)
                    tip_pt = hand_pt + (direction / norm * hand_length if norm > 1e-6 else onp.array([hand_length, 0, 0]))
                hand_cyl = _cylinder_between(
                    hand_pt, tip_pt,
                    radius=hand_thickness,
                    color=(255, 80, 0, 220),
                )
                if hand_cyl is not None:
                    server.scene.add_mesh_trimesh("/arm_hand", hand_cyl)
            # Change grab sphere color at grab_idx
            if grab_sphere is not None and grab_idx is not None:
                if tstep >= grab_idx:
                    grab_sphere.color = (50, 255, 50)  # Green: grab frame reached
                else:
                    grab_sphere.color = (255, 50, 50)  # Red: not yet reached

            skeleton = [
                [-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]
            ]
            # --- End of dummy data ---

            # Let's assume 'tstep' is defined elsewhere in your loop
            current_smpl_keypoints = smpl_keypoints[min(tstep, len(smpl_keypoints) - 1)]

            # Prepare points for the line segments in (N, 2, 3) format
            line_segment_points_list = []
            for bone in skeleton:
                idx0, idx1 = bone
                # Skip if the first index is -1
                if idx0 == -1:
                    continue
                # Ensure indices are within bounds
                if 0 <= idx0 < len(current_smpl_keypoints) and 0 <= idx1 < len(current_smpl_keypoints):
                    start_point = current_smpl_keypoints[idx0]
                    end_point = current_smpl_keypoints[idx1]
                    line_segment_points_list.append([start_point, end_point])
                else:
                    print(f"Warning: Bone {bone} has out-of-bounds indices for current_smpl_keypoints with shape {current_smpl_keypoints.shape}")

            if line_segment_points_list:
                # Convert to NumPy array of shape (N, 2, 3)
                points_for_lines = onp.array(line_segment_points_list)
                num_lines = len(points_for_lines)

                # Prepare colors for the line segments (N, 2, 3)
                # Let's make each segment uniformly blue. So both endpoints of a segment are blue.
                # RGB color for blue
                blue_color = onp.array([0, 0, 255])
                # Create an array (N, 2, 3) where each [start_color, end_color] pair is [blue, blue]
                colors_for_lines = onp.tile(blue_color, (num_lines, 2, 1))


                server.scene.add_line_segments(
                    name="/target_skeleton_segments", # Unique name
                    points=points_for_lines,
                    colors=colors_for_lines, # (N, 2, 3) array, colors for each endpoint
                    line_width=3.0, # As per your example
                )


            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(smpl_keypoints[min(tstep, len(smpl_keypoints) - 1)]),
                onp.array((0, 0, 255))[None].repeat(22, axis=0),
                point_size=0.01,
            )

        time.sleep(1./20.)




if __name__ == "__main__":

    # use with dreamcontrol env

    import argparse
    parser = argparse.ArgumentParser(description="Visualize pick motions")
    parser.add_argument("--id", type=int, default=10, help="Motion index to visualize")
    parser.add_argument("--ret_or_ref", type=int, choices=[1, 2], default=1, help="1=retargeted (Pick_sim1), 2=refined (Pick_sim2)")
    parser.add_argument("--segments", action="store_true", default=False, help="Show capsule arm segments instead of spheres")
    args = parser.parse_args()

    # --- Arm capsule visualization tuning (active when --segments is set) ---
    # Lengths and thicknesses mirror the values in refine_motions_al.py capsule_collision_cost calls.
    FOREARM_LENGTH            = 0.25   # metres: wrist-pitch link → hand origin
    FOREARM_THICKNESS         = 0.03   # metres: forearm capsule radius
    HAND_LENGTH               = 0.15   # metres: hand origin → fingertip (= HAND_TIP_OFFSET)
    HAND_THICKNESS            = 0.04   # metres: hand capsule radius
    COLLISION_CYLINDER_RADIUS = 0.05   # metres: grab-site obstacle cylinder (= R_OBSTACLE)

    data_file_id = str(args.id)
    ret_or_ref = str(args.ret_or_ref)

    human_data_path = "./sample/Pick_sim_hum/" + data_file_id + ".pkl"

    if args.ret_or_ref == 1:
        g1_data_path = "./sample/Pick_sim1/" + data_file_id + ".pkl"
    else:
        g1_data_path = "./sample/Pick_sim" + ret_or_ref + "/" + data_file_id + "_n.pkl"
    print("Right script?")
    main(
        show_segments=args.segments,
        forearm_length=FOREARM_LENGTH,
        forearm_thickness=FOREARM_THICKNESS,
        hand_length=HAND_LENGTH,
        hand_thickness=HAND_THICKNESS,
        collision_cylinder_radius=COLLISION_CYLINDER_RADIUS,
    )


