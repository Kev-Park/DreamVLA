"""Humanoid Retargeting

Retarget motion to G1 humanoid, with scene contacts (keep feet close to contact points, while avoiding world-collisions).
"""

import time

import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
import viser
from viser.extras import ViserUrdf

import torch
from isaac_utils.rotations import(
    quaternion_to_matrix
)
import pytorch_kinematics as pk2
import os
import sys

# Add parent directory to path to access ../retarget_helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retarget_helpers._utils import (
    SMPL_JOINT_NAMES,
    create_conn_tree,
    get_humanoid_retarget_indices_g1_27dof,
    RetargetingWeights,
    foot_detect,
    solve_retargeting,
    get_keypts,
    transform_keypts,
)
import pickle
import yourdfpy

VIS = False
FREEZE_FOR = 20
RIGHT_ELBOW_INDEX = 19
RIGHT_HAND_INDEX = 21
RIGHT_HAND_JOINT_INDEX = 38
TASK_NAME = "Button_Press_real1"

def main():
    """Main function for humanoid retargeting."""

    # urdf = load_robot_description("g1_description")
    urdf = yourdfpy.URDF.load('../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    urdf_path = '../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # Load source motion data:
    # - keypoints [N, 22, 3],
    # - left/right foot contact (boolean) 2 x [N],
    # - heightmap [H, W].
    # asset_dir = Path(__file__).parent / "retarget_helpers" / "humanoid" / "amass"
    DATA = onp.load('results.npy', allow_pickle=True).item()
    hints = DATA['hint'][:,1:]
    smpl_keypoints_ = DATA['motion'].transpose((0,3,1,2)) # Shape: (k, N, 22, 3)
    rot_mat = onp.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])
    # Rotate the smpl keypoints by rot_mat
    smpl_keypoints = onp.einsum('ij, ankj -> anki', rot_mat, smpl_keypoints_.copy()) # Shape: (k, N, 22, 3)

    is_left_foot_contact, is_right_foot_contact = foot_detect(smpl_keypoints, thres=0.002)
    smpl_keypoints = smpl_keypoints[:,1:]
    heightmap = onp.zeros((100, 100), dtype=onp.float32)  # Dummy heightmap, replace with actual data if available.
    
    num_motions = smpl_keypoints.shape[0]
    num_timesteps = smpl_keypoints.shape[1]
    
    
    assert smpl_keypoints.shape == (num_motions, num_timesteps, 22, 3)
    assert is_left_foot_contact.shape == (num_motions, num_timesteps,)
    assert is_right_foot_contact.shape == (num_motions, num_timesteps,)
    
    heightmap = pk.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.01, 1.0]),
        height_data=heightmap,
    )

    # Get the left and right foot keypoints, projected on the heightmap.
    left_foot_keypoint_idx = SMPL_JOINT_NAMES.index("left_foot")
    right_foot_keypoint_idx = SMPL_JOINT_NAMES.index("right_foot")
    left_foot_keypoints = smpl_keypoints[..., left_foot_keypoint_idx, :].reshape(-1, 3)
    right_foot_keypoints = smpl_keypoints[..., right_foot_keypoint_idx, :].reshape(
        -1, 3
    )
    left_foot_keypoints = heightmap.project_points(left_foot_keypoints).reshape(
        num_motions, num_timesteps, 3)
    right_foot_keypoints = heightmap.project_points(right_foot_keypoints).reshape( 
        num_motions, num_timesteps, 3)

    
    smpl_joint_retarget_indices, g1_joint_retarget_indices = (
        get_humanoid_retarget_indices_g1_27dof()
    )

    smpl_mask = create_conn_tree(robot, g1_joint_retarget_indices)

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

    weights = pk.viewer.WeightTuner(
        server,
        RetargetingWeights(
            local_alignment=2.0,
            global_alignment=1.0,
            floor_contact=1.0,
            root_smoothness=1.0,
            foot_skating=1.0,
            world_collision=1.0,
        ),  # type: ignore
    )

    Ts_world_roots, jointss = [], []

    def generate_trajectory():
        nonlocal Ts_world_roots, jointss
        for i in range(num_motions):
            print("Processing motion", i + 1, "of", num_motions)
            hint = hints[i]
            pick_point = jnp.array([0., 0., 0.])  # Default pick point if no hint is available.
            for joint_pos in hint:
                right_hand_pos = joint_pos[RIGHT_HAND_INDEX]
                if right_hand_pos[2] > 0. :
                    pick_point = jnp.array(right_hand_pos)
                    break
                
            Ts_world_root, joints = solve_retargeting(
                robot=robot,
                robot_coll=robot_coll,
                target_keypoints=smpl_keypoints[i],
                is_left_foot_contact=is_left_foot_contact[i],
                is_right_foot_contact=is_right_foot_contact[i],
                left_foot_keypoints=left_foot_keypoints[i],
                right_foot_keypoints=right_foot_keypoints[i],
                smpl_joint_retarget_indices=smpl_joint_retarget_indices,
                g1_joint_retarget_indices=g1_joint_retarget_indices,
                smpl_mask=smpl_mask,
                heightmap=heightmap,
                weights=weights.get_weights(),  # type: ignore
                pick_point=pick_point,  # Use pick point to create table constraints
            )
            Ts_world_roots.append(Ts_world_root)
            jointss.append(joints)
    
        
    generate_trajectory()

    pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())
    # assert Ts_world_root is not None and joints is not None
    for i in range(num_motions):
        hint = hints[i]
        
        # Find closest pose to righ_hand_pos
        
        global_position = torch.tensor(onp.array(Ts_world_roots[i].translation())).float().clone()
        global_orientation = torch.tensor(onp.array(Ts_world_roots[i].rotation().wxyz)).float().clone()
        jointss[i] = torch.tensor(onp.array(jointss[i])).float()
        
        joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        
        keypts = get_keypts(jointss[i], joint_names , pk2_robot=pk2_robot)
        # exit(0)
        global_keypts = transform_keypts(keypts, global_orientation, global_position)
        pick_t = onp.argmax(global_keypts[:,RIGHT_HAND_JOINT_INDEX,0])
        print("Pick time:", pick_t, "keypt:", smpl_keypoints_[i,pick_t,RIGHT_HAND_INDEX])
        
        # freeze for a few frames after pick_t
        min_z = torch.min(global_keypts[:,:,2],dim=1).values
        grab_pos = global_keypts[pick_t, RIGHT_HAND_JOINT_INDEX, :]
        global_position[:, 2] -= min_z
        grab_pos[2] -= min_z[pick_t]
        motion_len = jointss[0].shape[0]
        
        global_position[pick_t+FREEZE_FOR:] = global_position[pick_t:motion_len-FREEZE_FOR].clone()
        global_orientation[pick_t+FREEZE_FOR:] = global_orientation[pick_t:motion_len-FREEZE_FOR].clone()
        jointss[i][pick_t+FREEZE_FOR:] = jointss[i][pick_t:motion_len-FREEZE_FOR].clone()
        
        global_position[pick_t:pick_t+FREEZE_FOR] = global_position[pick_t]
        global_orientation[pick_t:pick_t+FREEZE_FOR] = global_orientation[pick_t]
        jointss[i][pick_t:pick_t+FREEZE_FOR] = jointss[i][pick_t]
        
        print(i, pick_t, grab_pos)
        global_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(global_orientation)),jnp.array(global_position))
        os.makedirs("../"+TASK_NAME, exist_ok=True)
        with open("../"+TASK_NAME+ "/" + str(i) + ".pkl", "wb") as f:
            pickle.dump({"global_pose": global_pose , "joints": jointss[i], "global_position": global_position, "hit_pos": grab_pos , "grab_idx": pick_t}, f)
    
    if VIS :
        while True:
            with server.atomic():
                if playing.value:
                    timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                tstep = timestep_slider.value
                base_frame.wxyz = onp.array(Ts_world_roots[0].wxyz_xyz[tstep][:4])
                base_frame.position = onp.array(Ts_world_roots[0].wxyz_xyz[tstep][4:])
                urdf_vis.update_cfg(onp.array(jointss[0][tstep]))

                skeleton = [
                    [-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                    [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]
                ]
                # --- End of dummy data ---

                # Let's assume 'tstep' is defined elsewhere in your loop
                current_smpl_keypoints = smpl_keypoints[tstep] # Replace with smpl_keypoints[tstep]

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


                    server.add_line_segments(
                        name="/target_skeleton_segments", # Unique name
                        points=points_for_lines,
                        colors=colors_for_lines, # (N, 2, 3) array, colors for each endpoint
                        line_width=3.0, # As per your example
                    )


                server.scene.add_point_cloud(
                    "/target_keypoints",
                    onp.array(smpl_keypoints[tstep]),
                    onp.array((0, 0, 255))[None].repeat(22, axis=0),
                    point_size=0.01,
                )

            time.sleep(0.2)



if __name__ == "__main__":
    main()
