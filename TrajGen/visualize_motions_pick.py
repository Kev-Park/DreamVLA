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



def main():
    
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
    global_keypts = transform_keypts(torch.tensor(keypts), torch.tensor(orientations), torch.tensor(positions)).numpy()
    
    # Create a sphere at the grab position
    sphere = trimesh.creation.icosphere(radius=0.05)
    transform = onp.eye(4)
    transform[:3, 3] = [global_keypts[-1, -3, 0] + 0.03 + 0.25, global_keypts[-1, -3, 1], global_keypts[-1, -3, 2]/2.-0.05]
    sphere.apply_transform(transform)
    print(transform[:3, 3])
    # Add the sphere to Viser
    server.scene.add_mesh_trimesh(
        name="my_sphere",
        mesh=sphere,
    )


        # Define cuboid dimensions (width, height, depth)
    width, height, depth = 0.5, global_keypts[34, -11, 2]-0.1, 0.5 #changed 65 to 34 in next parts due to clipped video

    # Create a cuboid using Trimesh
    cuboid = trimesh.creation.box(extents=(width, depth, height))

    # gen_button = server.gui.add_button("Retarget!")
    transform = onp.eye(4)
    transform[:3, 3] = [global_keypts[34, -3, 0] +0.1+ 0.25, global_keypts[34, -3, 1], global_keypts[34, -3, 2]/2.-0.05]
    cuboid.apply_transform(transform)
    print(transform[:3, 3])
    # Add the cuboid to Viser
    # server.scene.add_mesh_trimesh(
    #     name="my_cuboid1",
    #     mesh=cuboid,
    #     # wireframe=False,  # Set to True if you want to see just the wireframe
    #     # color=(0.2, 0.6, 0.9, 1.0),  # RGBA
    # )
    
    print(positions)
    print("Started?")
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


            
            # Visualize right hand position (link index 38 = right_rubber_hand)
            server.scene.add_point_cloud(
                "/right_hand",
                global_keypts[tstep, 38:39, :],  # shape (1, 3)
                onp.array([[255, 0, 0]]),          # red
                point_size=0.05,
            )
            # Visualize right wrist position (link index 36 = right_wrist_pitch_link)
            server.scene.add_point_cloud(
                "/right_wrist",
                global_keypts[tstep, 36:37, :],  # shape (1, 3)
                onp.array([[0, 0, 255]]),          # blue
                point_size=0.05,
            )

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
    args = parser.parse_args()

    data_file_id = str(args.id)
    ret_or_ref = str(args.ret_or_ref)

    human_data_path = "./sample/Pick_sim_hum/" + data_file_id + ".pkl"
    # g1_data_path = "000332.pkl"

    if args.ret_or_ref == 1:
        g1_data_path = "./sample/Pick_sim1/" + data_file_id + ".pkl"
    else:
        g1_data_path = "./sample/Pick_sim" + ret_or_ref + "/" + data_file_id + "_n.pkl"
    #g1_data_path = "follow1/0.pkl"
    print("Right script?")
    main()


