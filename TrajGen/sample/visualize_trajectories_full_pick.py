"""Full trajectory visualization: G1 robot + SMPL skeleton + table cuboid + grab point.

Usage (from TrajGen/sample/):
    python visualize_trajectories_full.py --folder Pick_sim2 --id 0_n
    python visualize_trajectories_full.py --folder Pick_sim1 --id 42
"""

import time
import os
import re

import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import pickle
import yourdfpy
import trimesh
import argparse


# SMPL skeleton connectivity (parent → child pairs for 22 joints)
SMPL_SKELETON = [
    [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
    [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21],
]

# Table dimensions (from refine_motions.py)
TABLE_WIDTH = 1.0   # x extent (depth of table surface)
TABLE_DEPTH = 2.0   # y extent (width of table surface)
TABLE_HEIGHT = 0.86  # z extent (OFFSET_Z in refine_motions.py)
# Note: refine_motions.py locally adds WRIST_TO_COLLISION (+0.35) to grab_pos
# then does grab_pos[0] + 0.5 + OFFSET_X (-0.35).  Since the pkl stores the
# raw wrist position (no collision offset), the +0.35 and -0.35 cancel out,
# so we just use grab_pos[0] + 0.5 directly for the table centre.


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_pkl_path(folder, id_str):
    """Try to find the .pkl file, handling both 'id' and 'id_n' formats."""
    direct = os.path.join(folder, f"{id_str}.pkl")
    if os.path.exists(direct):
        return direct
    # If --id doesn't include _n, try appending it
    with_n = os.path.join(folder, f"{id_str}_n.pkl")
    if os.path.exists(with_n):
        return with_n
    raise FileNotFoundError(
        f"Could not find {direct} or {with_n}"
    )


def extract_motion_index(id_str):
    """Extract the numeric motion index from an id like '42' or '42_n'."""
    match = re.match(r"(\d+)", id_str)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract motion index from id '{id_str}'")


def resolve_results_path(folder):
    """Derive the path to results.npy from the folder name.
    
    Pick_sim1 or Pick_sim2 → Pick_sim/results.npy
    Button_Press_sim1 → Button_Press_sim/results.npy
    """
    base = re.sub(r"\d+$", "", folder)
    results_path = os.path.join(base, "results.npy")
    if os.path.exists(results_path):
        return results_path
    raise FileNotFoundError(
        f"Could not find results.npy at {results_path}. "
        f"Make sure you run this from TrajGen/sample/."
    )


def load_smpl_keypoints(results_path, motion_idx):
    """Load and transform SMPL keypoints for a specific motion index."""
    data = onp.load(results_path, allow_pickle=True).item()
    # motion shape: (batch, joints, channels, timesteps) — take first 3 channels (xyz)
    motion = data["motion"]
    keypoints_raw = motion.transpose((0, 3, 1, 2))  # → (batch, T, 22, channels)
    # Take only xyz (first 3 of 6 channels if cont6d, or all 3 if already 3D)
    keypoints_raw = keypoints_raw[..., :3]

    # Apply coordinate rotation (same as retarget.py)
    rot_mat = onp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    keypoints = onp.einsum("ij, ankj -> anki", rot_mat, keypoints_raw.copy())

    # Trim first frame (retarget.py does smpl_keypoints[:,1:])
    keypoints = keypoints[:, 1:]

    return keypoints[motion_idx]  # (T, 22, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize G1 robot + SMPL skeleton + table + grab point"
    )
    parser.add_argument("--folder", type=str, default="Pick_sim2",
                        help="Folder containing .pkl files (e.g. Pick_sim1, Pick_sim2)")
    parser.add_argument("--id", type=str, default="0",
                        help="Motion id (e.g. '0', '42', '42_n')")
    parser.add_argument("--no-table", action="store_true",
                        help="Hide the table cuboid")
    parser.add_argument("--no-smpl", action="store_true",
                        help="Hide the SMPL skeleton overlay")
    args = parser.parse_args()

    # --- Load G1 robot trajectory (.pkl) ---
    pkl_path = resolve_pkl_path(args.folder, args.id)
    print(f"Loading robot trajectory: {pkl_path}")
    g1_data = load_pickle(pkl_path)

    global_pose = g1_data["global_pose"]
    joints = g1_data["joints"]
    grab_pos = onp.array(g1_data.get("grab_pos", onp.zeros(3)))
    grab_idx = g1_data.get("grab_idx", 0)
    num_robot_frames = joints.shape[0]

    positions = onp.array(global_pose.translation())
    orientations = onp.array(global_pose.rotation().wxyz)

    # --- Load SMPL keypoints (results.npy) ---
    smpl_keypoints = None
    if not args.no_smpl:
        try:
            motion_idx = extract_motion_index(args.id)
            results_path = resolve_results_path(args.folder)
            print(f"Loading SMPL motion {motion_idx} from: {results_path}")
            smpl_keypoints = load_smpl_keypoints(results_path, motion_idx)
            num_smpl_frames = smpl_keypoints.shape[0]
            print(f"  Robot frames: {num_robot_frames}, SMPL frames: {num_smpl_frames}")
        except (FileNotFoundError, IndexError) as e:
            print(f"Warning: Could not load SMPL data: {e}")
            print("  Continuing without SMPL skeleton overlay.")
            smpl_keypoints = None

    # Compute frame offset (Pick_sim2 has 20 prepended frames)
    if smpl_keypoints is not None:
        frame_offset = max(0, num_robot_frames - smpl_keypoints.shape[0])
    else:
        frame_offset = 0

    # --- Set up viser scene ---
    urdf = yourdfpy.URDF.load(
        "../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
    )

    server = viser.ViserServer()

    # Ground plane
    heightmap_data = onp.zeros((1000, 1000), dtype=onp.float32)
    heightmap = pk.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.01, 1.0]),
        height_data=heightmap_data,
    )
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

    # G1 robot
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Table cuboid
    if not args.no_table and onp.any(grab_pos != 0):
        cuboid = trimesh.creation.box(
            extents=(TABLE_WIDTH, TABLE_DEPTH, TABLE_HEIGHT)
        )
        transform = onp.eye(4)
        # Front edge at grab_pos[0], extends TABLE_WIDTH behind it
        transform[:3, 3] = [
            grab_pos[0] + TABLE_WIDTH / 2.0,
            0.0,
            TABLE_HEIGHT / 2.0,
        ]
        cuboid.apply_transform(transform)
        cuboid.visual.face_colors = [180, 140, 100, 120]  # Semi-transparent wood color
        server.scene.add_mesh_trimesh("/table", cuboid)
        print(f"  Table placed at x={transform[0, 3]:.2f}, z={TABLE_HEIGHT:.2f}")

    # Grab point sphere
    if onp.any(grab_pos != 0):
        server.scene.add_icosphere(
            "/grab_point",
            radius=0.03,
            color=(255, 50, 50),
            position=onp.array(grab_pos, dtype=onp.float64),
        )
        print(f"  Grab point at {grab_pos}, grab frame: {grab_idx}")

    # GUI controls
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider(
        "timestep", 0, num_robot_frames - 1, 1, 0
    )
    show_smpl = server.gui.add_checkbox("show_smpl", smpl_keypoints is not None)
    smpl_opacity = server.gui.add_slider("smpl_opacity", 0.0, 1.0, 0.01, 0.6)

    print(f"\nVisualization ready — open http://localhost:8080")
    print(f"  Frame offset (SMPL → robot): {frame_offset}")

    # --- Animation loop ---
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_robot_frames
            tstep = timestep_slider.value

            # Update G1 robot
            base_frame.wxyz = orientations[tstep]
            base_frame.position = positions[tstep] + onp.array([0, 0, 0.035])
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            # Update SMPL skeleton
            if smpl_keypoints is not None and show_smpl.value:
                smpl_t = tstep - frame_offset
                if 0 <= smpl_t < smpl_keypoints.shape[0]:
                    kp = smpl_keypoints[smpl_t]

                    # Joint point cloud
                    alpha = smpl_opacity.value
                    blue = int(255 * alpha)
                    color = onp.array([0, 0, blue])
                    server.scene.add_point_cloud(
                        "/smpl_keypoints",
                        kp,
                        onp.tile(color, (22, 1)),
                        point_size=0.015,
                    )

                    # Skeleton line segments
                    segs = []
                    for a, b in SMPL_SKELETON:
                        if a < len(kp) and b < len(kp):
                            segs.append([kp[a], kp[b]])
                    if segs:
                        server.scene.add_line_segments(
                            "/smpl_skeleton",
                            onp.array(segs),
                            colors=(0, 0, blue),
                            line_width=2.0,
                        )
                else:
                    # Outside SMPL range — hide skeleton
                    server.scene.add_point_cloud(
                        "/smpl_keypoints",
                        onp.zeros((1, 3)),
                        onp.zeros((1, 3), dtype=onp.uint8),
                        point_size=0.001,
                    )

        time.sleep(1.0 / 20.0)


if __name__ == "__main__":
    main()
