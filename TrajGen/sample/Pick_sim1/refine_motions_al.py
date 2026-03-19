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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load URDF and create kinematic chain
urdf_path = '../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=DEVICE)
pkl_paths = '*.pkl'
pkl_paths = glob.glob(pkl_paths)
SMOOTH_AMT = 20
PAUSE_AMT = 10
INTERP_AMT = 10
WRIST_TO_COLLISION = 0.35
VISUALIZE = False
OFFSET_Z = 0.86
OFFSET_X = -0.35
HAND_TIP_OFFSET = 0.15
TIP_SPEED_WINDOW = 50
SAVE_DIR = "../Pick_sim2/"
TRAJ_FPS_DEFAULT = 20.0

# Right-arm hard speed limits (rad/s)
RIGHT_ARM_SPEED_LIMITS = {
    "right_shoulder_pitch_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "right_elbow_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "right_wrist_pitch_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
}
DOF_SPEED_CONSTRAINT_TOL = 1e-3  # rad/s

# Updated per motion file (fallback to TRAJ_FPS_DEFAULT)
traj_fps_hz = TRAJ_FPS_DEFAULT

# --- Augmented Lagrangian settings ---
TABLE_CONSTRAINT_TOL = 1e-5   # 0.01 mm  – very tight: any tip penetration > this triggers AL update
AL_RHO_INIT       = 50.0      # initial quadratic penalty weight
AL_RHO_GROWTH     = 10.0      # penalty multiplier each outer iteration
AL_RHO_MAX        = 1e8       # cap so gradients don't explode for quadratic hard optimization
AL_OUTER_ITERS    = 40        # max AL outer iterations
AL_INNER_ITERS    = 300       # Adam steps per outer iteration

# Written by compute_cost every forward pass; read by the AL outer loop.
_last_g_t:         torch.Tensor | None = None
_last_g_cap_wrist: torch.Tensor | None = None
_last_g_dof_speed: torch.Tensor | None = None

def compute_cost(joint_angles, trans, quats, offset_x=OFFSET_X, offset_z=OFFSET_Z, debug=False,
                 lambda_table: torch.Tensor | None = None,
                 lambda_wrist: torch.Tensor | None = None,
                 lambda_dof_speed: torch.Tensor | None = None,
                 rho_al: float = AL_RHO_INIT):
    global _last_g_t, _last_g_cap_wrist, _last_g_dof_speed

    def al_penalty(g, lam_vec=None):
        """Augmented-Lagrangian penalty for a hard constraint g >= 0.
        lam_vec: dedicated dual variable tensor; auto-sliced to len(g)."""
        if lam_vec is not None:
            lam = lam_vec[:len(g)]
            return lam * g + (rho_al / 2.0) * g ** 2
        return (rho_al / 2.0) * g ** 2

    # L2 cost to target
    l2_cost = 0.*torch.nn.functional.mse_loss(joint_angles, target_joint_angles[:, active_joint_ids])

    # [ABS] Right-arm DOF hard speed limits (rad/s), enforced via AL.
    # joint_angles are in rad/frame, so convert with current trajectory FPS.
    dt = 1.0 / float(traj_fps_hz)
    joint_speed_rad_s = torch.abs(joint_angles[1:] - joint_angles[:-1]) / dt  # (T-1, 7)
    dof_limit_vec = torch.tensor(
        [RIGHT_ARM_SPEED_LIMITS[name] for name in active_joint_names],
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    ).unsqueeze(0)  # (1, 7)
    g_dof_speed = torch.relu(joint_speed_rad_s - dof_limit_vec)  # (T-1, 7), hard violation in rad/s
    g_dof_speed_flat = g_dof_speed.reshape(-1)
    _last_g_dof_speed = g_dof_speed_flat.detach()
    dof_speed_hard_cost = torch.mean(al_penalty(g_dof_speed_flat, lam_vec=lambda_dof_speed))
    
    # Forward kinematics to get link positions
    q_dict = {name: joint_angles[:, i] for i, name in enumerate( active_joint_names ) }
    for id in inactive_joint_ids:
        name = joint_names[id]
        q_dict[name] = target_joint_angles[:, id]
    fk_results = chain.forward_kinematics(q_dict)
    # Collect all link positions
    cost2 = torch.zeros(joint_angles.shape[0], device=joint_angles.device)
    rot_matrix = quaternion_to_matrix(quats)

    # Iterate over links and apply costs based on link positions and orientations
    for i, (link_name, tf) in enumerate(fk_results.items()):
    
        def capsule_collision_cost(origin_world, rot_world, segment_length, segment_thickness, collision_site, r_obstacle=0.05):
            """Arm capsule vs infinite vertical cylinder obstacle.

            The obstacle is an infinite vertical cylinder centred at
            collision_site XY with radius r_obstacle.  For an infinite-Z
            cylinder the minimum 3D capsule distance reduces to the 2D
            segment-to-point distance in XY.

            The arm capsule runs from origin_world along the link's local
            x-axis for segment_length, with radius segment_thickness.
            Returns relu(r_total - d_xy) per frame.
            """
            r_total = segment_thickness + r_obstacle

            # Arm capsule axis in world frame (local x-axis of link frame)
            axis  = rot_world[:, :, 0]                            # (N, 3)
            P1    = origin_world                                   # (N, 3) proximal end
            P2    = origin_world + axis * segment_length           # (N, 3) distal end

            # Project to XY — infinite-Z cylinder reduces to point in 2D
            A  = P1[:, :2]                                         # (N, 2)
            B  = P2[:, :2]                                         # (N, 2)
            P  = collision_site[:2].unsqueeze(0)                   # (1, 2)

            AB  = B - A                                            # (N, 2)
            AP  = P - A                                            # (N, 2)
            ab2 = (AB * AB).sum(dim=1).clamp(min=1e-8)            # (N,)
            t   = ((AP * AB).sum(dim=1) / ab2).clamp(0., 1.)      # (N,) parameter on segment

            closest = A + t.unsqueeze(1) * AB                     # (N, 2) nearest point on segment
            d_xy    = torch.norm(P - closest, dim=1)               # (N,) distance to cylinder axis

            return torch.relu(r_total - d_xy)                      # (N,) penetration depth

        if i == 36: # right wrist pitch link (blue dot in viser)
            pos = tf.get_matrix()[:,:3,3]
            pos_ref = fk_results_ref[link_name].get_matrix()[:,:3,3]

            cost2 += 0.*torch.mean(torch.norm(pos[1:] - pos[:-1], dim=1))
            trans_with_z = trans + torch.tensor([0., 0., 0.035], device=trans.device)
            transformed_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_with_z
            transformed_keypts_ref = torch.bmm(pos_ref.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_with_z
            
            # displacement per pair of frames
            rel_dists = torch.norm(transformed_keypts[1:] - transformed_keypts[:-1], dim=1)

            # GENERAL COSTS
            # [ABS] Penalize wrist speed changes (acceleration) and high speed relative to own trajectory
            wrist_accel = rel_dists[1:] - rel_dists[:-1]
            cost2[1:] += 50*torch.abs(rel_dists)**2 # L2 speed magnitude (dominates large speeds)
            cost2[1:-1] += 50*(wrist_accel)**2  # acceleration
            wrist_jerk = wrist_accel[1:] - wrist_accel[:-1]
            cost2[2:-1] += 50. * wrist_jerk ** 2

            # PRE-APPROACH COSTS

            # Use reference tracking costs in the initial approach (good motion sentiment reference)
            ref = False
            if ref:
                # Jerkiness penalty relative to references
                max_ref_dists = torch.max(ref_dists) # get max speed for normalizing
                ja_diff = 0.03*torch.sum(torch.abs(joint_angles[1:] - joint_angles[:-1]), dim=1) # get total joint change * 0.03 (total jerkiness)
                cost2[:-1] += ja_diff * (1.2-ref_dists/max_ref_dists) # normalize velocities; make max penalty 0.2, multiply by jerkiness to penalize low speed jerkiness

                # [REF] IMPORTANT SECTION - Penalize wrist speed deviation from reference retargeted motion
                vals = rel_dists[:grab_idx+2] - ref_dists[:grab_idx+2]
                cost2[1:grab_idx+2] += torch.abs(vals[1:] - vals[:-1])  # get difference in velocity errors (acceleration?)
                cost2[:grab_idx+2] += torch.abs(vals)                     # velocity error
                cost2[:grab_idx+2] += 10*vals**2                          # L2 speed deviation (dominates large errors)


            # RECOMMENDED COSTS
            # [REF] 1-step and 2-step velocity magnitude matching — penalise deviation of speed from reference
            # 1-step: speed t→t+1
            # opt_speed_1 = torch.norm(transformed_keypts[1:] - transformed_keypts[:-1], dim=1)      # (N-1,)
            # ref_speed_1 = torch.norm(transformed_keypts_ref[1:] - transformed_keypts_ref[:-1], dim=1)  # (N-1,)
            # cost2[1:] += 10 * (opt_speed_1 - ref_speed_1) ** 2

            # # 2-step: speed t→t+2 (captures stride-level speed)
            # opt_speed_2 = torch.norm(transformed_keypts[2:] - transformed_keypts[:-2], dim=1)      # (N-2,)
            # ref_speed_2 = torch.norm(transformed_keypts_ref[2:] - transformed_keypts_ref[:-2], dim=1)  # (N-2,)
            # cost2[2:] += 10 * (opt_speed_2 - ref_speed_2) ** 2


            # [REF] Laziness: penalize deviation from rest pose, decaying toward grab_idx
            rest_pose = torch.tensor(init_joint_angles, device=DEVICE)[active_joint_ids]
            # Clamp effective end to actual sequence length — grab_idx can exceed N for some pkl files.
            laziness_end = min(max(grab_idx - 40, 0), joint_angles.shape[0])
            if laziness_end > 0:
                laziness_weight = torch.linspace(1.0, 0.0, laziness_end, device=DEVICE)
                cost2[:laziness_end] += laziness_weight * torch.sum(torch.abs(joint_angles[:laziness_end] - rest_pose), dim=1)

            # INTER-APPROACH COSTS

            # [ABS] Forearm capsule collision with grab obstacle — hard constraint via AL
            # The forearm capsule runs from the wrist origin (link 36) toward the hand origin (link 38).
            # We resolve the axis lazily inside compute_cost by storing wrist pos and computing
            # the direction to the hand link when i==38.  For the wrist pass we just stash the
            # transformed wrist position so the hand pass (i==38) can pick it up.
            _wrist_pos_world = transformed_keypts  # (N, 3) — used by i==38 for forearm axis

            # POST-APPROACH COSTS

            #cost2[grab_idx:] += 10.*(joint_angles[grab_idx:, -6]+0.15)*(joint_angles[grab_idx:, -6]>-0.15)  # [ABS] Penalize right shoulder exceeding -0.15 (absolute joint limit)

            
            #transformed_keypts_ref[grab_idx:, 2] = torch.maximum(transformed_keypts_ref[grab_idx:, 2], torch.tensor(offset_z, device=DEVICE)) # freeze reference height
            #cost2[grab_idx-15:grab_idx] += torch.norm(transformed_keypts[grab_idx-15:grab_idx] - transformed_keypts_ref[grab_idx-15:grab_idx], dim=1, p=2) # penalize distance from reference after grab

        elif i == 38: # right hand link (red dot in viser)
            rot_mat = tf.get_matrix()[:,:3,:3]
            rot_mat = torch.bmm(rot_matrix, rot_mat)
            rot_mat_ref = torch.tensor([[1, 0, 0], 
                                        [0, 1, 0], 
                                        [0, 0, 1]], dtype=torch.float32, device=DEVICE).T
            rot_mat = torch.bmm(rot_mat, rot_mat_ref.unsqueeze(0).expand(rot_mat.shape[0], -1, -1))
            angle = torch.acos(torch.clamp((rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2] - 1) / 2, -0.999, .999))
            #cost2 += 0.3*angle  # [REF] Penalize hand orientation deviating from identity (palm-down)

            # Get hand position and rotation in world frame
            hand_tf = tf.get_matrix()                   # (N, 4, 4)
            hand_pos = hand_tf[:, :3, 3]                # joint origin (N, 3)
            hand_rot = hand_tf[:, :3, :3]               # hand local rotation (N, 3, 3)
            local_tip = torch.tensor([HAND_TIP_OFFSET, 0., 0.], device=DEVICE)
            tip_pos = hand_pos + torch.bmm(hand_rot, local_tip.view(1, 3, 1).expand(hand_rot.shape[0], -1, -1)).squeeze(-1)
            trans_with_z = trans + torch.tensor([0., 0., 0.035], device=trans.device)
            transformed_tip = torch.bmm(tip_pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z  # final hand position
            # Hand joint origin in world frame — forms the other edge of the swept quad
            transformed_hand_orig = torch.bmm(hand_pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z


            # INTER-APPROACH COSTS

            # [ABS] Forearm capsule collision — axis runs from wrist origin (link 36) toward hand
            # origin (link 38), matching the visual in visualize_motions_pick.py.  We use the
            # stashed _wrist_pos_world from the i==36 branch (already includes the 0.035 offset).
            ENABLE_WRIST_CAPSULE_COST = False
            if ENABLE_WRIST_CAPSULE_COST and '_wrist_pos_world' in dir():
                forearm_dir_world = transformed_hand_orig - _wrist_pos_world  # (N, 3)
                forearm_dist = torch.norm(forearm_dir_world, dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
                forearm_axis_world = forearm_dir_world / forearm_dist          # (N, 3) unit vector
                # Build a rotation-matrix-like (N, 3, 3) where the first column is our axis.
                forearm_rot_world = torch.zeros(forearm_axis_world.shape[0], 3, 3, device=DEVICE)
                forearm_rot_world[:, :, 0] = forearm_axis_world
                # Fill remaining columns with any orthonormal basis (only col-0 is used by capsule_collision_cost)
                perp = torch.zeros_like(forearm_axis_world)
                perp[:, 1] = 1.0
                dot = (forearm_axis_world * perp).sum(dim=1, keepdim=True)
                perp = perp - dot * forearm_axis_world
                perp_norm = torch.norm(perp, dim=1, keepdim=True).clamp(min=1e-8)
                perp = perp / perp_norm
                forearm_rot_world[:, :, 1] = perp
                forearm_rot_world[:, :, 2] = torch.linalg.cross(forearm_axis_world, perp)
                g_cap_wrist = capsule_collision_cost(
                    _wrist_pos_world, forearm_rot_world, segment_length=0.25, segment_thickness=0.03,
                    collision_site=capsule_obs_pos
                )
            else:
                g_cap_wrist = torch.zeros(transformed_hand_orig.shape[0], device=DEVICE)
            _last_g_cap_wrist = g_cap_wrist[:max(grab_idx-15, 1)].detach()
            cost2[:max(grab_idx-15, 1)] += al_penalty(g_cap_wrist[:max(grab_idx-15, 1)], lam_vec=lambda_wrist)

            # table collision with anti-tunneling costs (test later - can remove g_point?)
            def table_collision_cost(pts, orig_pts, gi):
                """Combined per-frame AL constraint g_t >= 0, shape (gi,).

                For each frame t in [0, gi):
                  g_t = max(point_depth_t, segment_penetration_length_t)

                point_depth_t              : min(relu(Δx), relu(Δz)) at frame t.
                segment_penetration_length_t : length in metres of the motion segment
                  [t → t+1] that lies inside the table zone (x > x_edge AND z < z_table),
                  computed over both the fingertip edge and the hand-origin edge and
                  taking the worse of the two.  Frame gi-1 has no outgoing segment so
                  only point depth applies there.

                Units: metres throughout — commensurable with TABLE_CONSTRAINT_TOL.
                Drop-in replacement: same (gi,) shape, same sign convention, same AL loop.
                """
                x_edge  = grab_pos[0] + offset_x
                z_table = offset_z
                eps     = 1e-8

                # ---- 1. Point-wise penetration depth (existing logic, unchanged) ----
                depth_x = torch.relu(pts[:gi, 0] - x_edge)
                depth_z = torch.relu(z_table - pts[:gi, 2])
                g_point = torch.minimum(depth_x, depth_z)            # (gi,)

                # ---- 2. Segment overlap length  (anti-tunneling) ----
                # Reuse the exact parametric logic from the original seg_overlap helper
                # but scale by segment length so the output is in metres.
                def _overlap_frac(a, b):
                    """Fraction of segment [a→b] (W,3) inside the table zone."""
                    ax, az = a[:, 0], a[:, 2]
                    bx, bz = b[:, 0], b[:, 2]
                    ddx = bx - ax
                    ddz = bz - az
                    # x: inside when x > x_edge
                    scx = torch.clamp((x_edge - ax) / (ddx + eps), 0., 1.)
                    sx0 = torch.where(ddx > 0, scx,                    torch.zeros_like(scx))
                    sx1 = torch.where(ddx > 0, torch.ones_like(scx),   scx)
                    sx0 = torch.where(ddx.abs() < eps,
                                      torch.where(ax > x_edge, torch.zeros_like(sx0), torch.ones_like(sx0)), sx0)
                    sx1 = torch.where(ddx.abs() < eps,
                                      torch.where(ax > x_edge, torch.ones_like(sx1),  torch.zeros_like(sx1)), sx1)
                    # z: inside when z < z_table  (opposite orientation to x: zone is "z small")
                    # ddz < 0: z falling, enters zone at scz → inside [scz, 1]
                    # ddz > 0: z rising, starts inside and exits at scz → inside [0, scz]
                    scz = torch.clamp((z_table - az) / (ddz + eps), 0., 1.)
                    sz0 = torch.where(ddz < 0, scz,                    torch.zeros_like(scz))
                    sz1 = torch.where(ddz < 0, torch.ones_like(scz),   scz)
                    sz0 = torch.where(ddz.abs() < eps,
                                      torch.where(az < z_table, torch.zeros_like(sz0), torch.ones_like(sz0)), sz0)
                    sz1 = torch.where(ddz.abs() < eps,
                                      torch.where(az < z_table, torch.ones_like(sz1),  torch.zeros_like(sz1)), sz1)
                    return torch.relu(torch.minimum(sx1, sz1) - torch.maximum(sx0, sz0))

                # gi-1 consecutive segments
                p0, p1 = pts[:gi - 1],      pts[1:gi]         # fingertip edge
                o0, o1 = orig_pts[:gi - 1], orig_pts[1:gi]   # hand-origin edge

                frac_tip  = _overlap_frac(p0, p1)              # (gi-1,)
                frac_orig = _overlap_frac(o0, o1)              # (gi-1,)

                # Scale each edge by its own arc length so the wrist-origin edge
                # (which can sweep a larger arc) isn't underweighted.
                g_seg_tip  = frac_tip  * torch.norm(p1 - p0, dim=1)
                g_seg_orig = frac_orig * torch.norm(o1 - o0, dim=1)
                g_seg      = torch.maximum(g_seg_tip, g_seg_orig)  # (gi-1,)

                # Last frame has no outgoing segment — pad with zero
                g_seg = torch.cat([g_seg, g_seg.new_zeros(1)]) # (gi,)

                return torch.maximum(g_point, g_seg)           # (gi,)  hard constraint

            # [ABS] Per-frame table collision — hard constraint via AL (g_t >= 0)
            # Enforce over the full trajectory.
            table_gi = transformed_tip.shape[0]
            g_t = table_collision_cost(transformed_tip, transformed_hand_orig, table_gi)
            _last_g_t = g_t.detach()          # expose to outer AL loop (no-graph copy) for hard optimization
            cost2[:table_gi] += al_penalty(g_t, lam_vec=lambda_table)


            # [ABS] Penalize hand-tip speed changes (acceleration), speed, and jerk
            tip_speed = torch.norm(transformed_tip[1:] - transformed_tip[:-1], dim=1)   # (N-1,) speed per frame
            cost2[1:] += 25*torch.abs(tip_speed)**2  # L2 speed magnitude (dominates large speeds)
            tip_accel = tip_speed[1:] - tip_speed[:-1]
            cost2[1:-1] += 25*(tip_accel)**2  # acceleration
            tip_jerk = tip_accel[1:] - tip_accel[:-1]
            cost2[2:-1] += 50. * tip_jerk ** 2


            # [SOFT] Hand-tip approach shaping in X:
            # Asymmetric X-distance cost: only penalize if hand is LEFT of grab point.
            # Allows rightward movement without penalty, prevents self-intersection from leftward drift.
            tip_x = transformed_tip[:grab_idx, 0]
            grab_x = grab_pos[0] + offset_x
            x_dist_left = torch.relu(grab_x - tip_x)  # (G,) Distance to the left; 0 if to the right
            approach_pen = x_dist_left ** 2

            n_trans = approach_pen.shape[0]
            if n_trans > 0:
                frame_ids = torch.arange(1, n_trans + 1, device=DEVICE, dtype=joint_angles.dtype)  # destination frames
                z_table = offset_z
                tip_above_mask = transformed_tip[:grab_idx, 2] >= z_table
                above_idx = torch.nonzero(tip_above_mask, as_tuple=False)
                if above_idx.numel() > 0:
                    taper_end = int(above_idx[0].item())
                else:
                    taper_end = max(grab_idx - 10, 1)
                taper_end = max(min(taper_end, n_trans), 1)
                taper_start = max(taper_end - 10, 1)
                taper = torch.ones(n_trans, device=DEVICE, dtype=joint_angles.dtype)
                taper_mask = frame_ids >= float(taper_start)
                taper_denom = max(taper_end - taper_start, 1)
                taper[taper_mask] = torch.clamp((float(taper_end) - frame_ids[taper_mask]) / float(taper_denom), min=0.0, max=1.0)

                HAND_APPROACH_SOFT_WEIGHT = 500.0
                cost2[:grab_idx] += HAND_APPROACH_SOFT_WEIGHT * approach_pen
        else:
            continue

    return l2_cost + torch.mean(cost2) + dof_speed_hard_cost

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

    target_trans = torch.tensor(np.array(motion_data['global_position']), dtype=torch.float32).to(DEVICE)
    target_quats = torch.tensor(np.array(motion_data['global_pose'].rotation().wxyz), dtype=torch.float32).to(DEVICE)

    # Target joint angles (example, replace with your actual target)
    target_joint_angles = torch.tensor(np.array(motion_data['joints']), dtype=torch.float32).to(DEVICE)
    grab_pos = torch.tensor(np.array(motion_data['grab_pos']), dtype=torch.float32).to(DEVICE)

    joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    init_joint_angles = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0., 0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.]
    inactive_joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
    active_joint_ids = [i for i, name in enumerate(joint_names) if name not in inactive_joint_names]
    active_joint_names = [name for name in joint_names if name not in inactive_joint_names]
    inactive_joint_ids = [i for i, name in enumerate(joint_names) if name in inactive_joint_names]
    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate( joint_names ) }
    fk_results_ref = chain.forward_kinematics(q_dict)
    # Initial guess for joint angles (can be zeros or random)
    joint_angles = torch.nn.Parameter(target_joint_angles[:, active_joint_ids].clone())  # Only optimize active joints
    trans = target_trans.clone()  # Translation offset (0.035 z added inside compute_cost)
    quats = target_quats.clone()  # Quaternion offset
    optimizer = optim.Adam([joint_angles], lr=0.001)
    grab_idx = motion_data["grab_idx"]
    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate( joint_names ) }
    fk_results = chain.forward_kinematics(q_dict)
    tf = fk_results["right_wrist_pitch_link"]
    pos = tf.get_matrix()[:,:3,3]
    rot_matrix = quaternion_to_matrix(quats)
    # import pdb; pdb.set_trace()
    # Apply the same 0.035 z-offset used inside compute_cost so capsule_obs_pos matches
    # the world-frame positions that the collision cost will actually operate on.
    trans_with_z_setup = trans + torch.tensor([0., 0., 0.035], device=trans.device)
    wrist_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_with_z_setup
    grab_pos = wrist_keypts[grab_idx].clone()
    grab_pos[0] += WRIST_TO_COLLISION  # kept for table_collision_cost x_edge

    # Capsule obstacle: use the ground-truth object position from the pkl, which is exactly
    # where visualize_motions_pick.py places its collision cylinder.
    # Only XY is used by capsule_collision_cost (infinite vertical cylinder), but we keep
    # the full 3D point so the debug pkl renders the cyan sphere at the right height too.
    capsule_obs_pos = torch.tensor(np.array(motion_data['grab_pos']), dtype=torch.float32).to(DEVICE)
    ref_dists = torch.norm(wrist_keypts[1:] - wrist_keypts[:-1], dim=1)

    # -----------------------------------------------------------------------
    # Augmented Lagrangian optimisation
    # Inner loop: Adam minimises  f(q) + λ·g + (ρ/2)·‖g‖²  over joint angles.
    # Outer loop: update multipliers λ ← max(0, λ + ρ·g)  and grow ρ.
    # All inner tensor ops are batched over the N-frame trajectory and run on
    # DEVICE (GPU when available) for full CUDA parallelism.
    # (All tensors are already on DEVICE from load time above.)
    # -----------------------------------------------------------------------
    table_gi = joint_angles.shape[0]
    lambda_table = torch.zeros(table_gi, device=DEVICE)   # table surface dual vars (full trajectory)
    wrist_gi = max(grab_idx - 15, 1)
    lambda_wrist  = torch.zeros(wrist_gi, device=DEVICE)  # wrist capsule dual vars (grab_idx-15 frames)
    lambda_dof_speed = torch.zeros((joint_angles.shape[0] - 1) * joint_angles.shape[1], device=DEVICE)
    # lambda_hand removed: hand collision is now a soft cost, not an AL hard constraint
    traj_fps_hz = float(motion_data.get("fps", motion_data.get("frame_rate", TRAJ_FPS_DEFAULT)))
    rho_al = AL_RHO_INIT

    converged = False
    for outer_iter in range(AL_OUTER_ITERS):
        # ---- inner minimisation with fixed dual vars and rho_al ----
        # Re-create Adam so stale first/second-moment estimates from the previous
        # (lighter-penalty) outer iteration don't corrupt the step direction now
        # that rho_al is much larger.
        optimizer = optim.Adam([joint_angles], lr=0.001)
        n_inner = AL_INNER_ITERS
        for i in range(n_inner):
            optimizer.zero_grad()
            cost = compute_cost(joint_angles, trans, quats,
                                lambda_table=lambda_table,
                                lambda_wrist=lambda_wrist,
                                lambda_dof_speed=lambda_dof_speed,
                                rho_al=rho_al)
            cost.backward()
            optimizer.step()

        # ---- read last constraint violations written by compute_cost ----
        g_curr       = _last_g_t          # shape (grab_idx,)
        g_curr_wrist = _last_g_cap_wrist  # shape (grab_idx,)
        g_curr_dof_speed = _last_g_dof_speed  # shape ((T-1)*7,)
        if g_curr is None:
            break

        # ---- dual (multiplier) update: λ ← max(0, λ + ρ·g) ----
        with torch.no_grad():
            lambda_table = torch.clamp(lambda_table + rho_al * g_curr, min=0.0)
            if g_curr_wrist is not None:
                lambda_wrist = torch.clamp(lambda_wrist + rho_al * g_curr_wrist, min=0.0)
            if g_curr_dof_speed is not None:
                lambda_dof_speed = torch.clamp(lambda_dof_speed + rho_al * g_curr_dof_speed, min=0.0)

        max_viol       = float(g_curr.max())
        max_viol_wrist = float(g_curr_wrist.max()) if g_curr_wrist is not None else 0.0
        max_viol_dof_speed = float(g_curr_dof_speed.max()) if g_curr_dof_speed is not None else 0.0
        print(f"[AL outer={outer_iter:02d}] table={max_viol:.2e}m  "
              f"wrist_cap={max_viol_wrist:.2e}m  dof_speed={max_viol_dof_speed:.2e}rad/s  "
              f"rho={rho_al:.1e}  cost={cost.item():.4f}")

        if (max_viol < TABLE_CONSTRAINT_TOL and
                max_viol_wrist < TABLE_CONSTRAINT_TOL and
                max_viol_dof_speed < DOF_SPEED_CONSTRAINT_TOL):
            print(f"  -> All constraints satisfied to table/wrist={TABLE_CONSTRAINT_TOL:.0e} m "
                f"and dof_speed={DOF_SPEED_CONSTRAINT_TOL:.0e} rad/s. Done.")
            converged = True
            break

        # ---- increase penalty for next outer iteration ----
        rho_al = min(rho_al * AL_RHO_GROWTH, AL_RHO_MAX)

    if not converged:
        print(f"[AL] Warning: did not converge within {AL_OUTER_ITERS} outer iterations. "
              f"Final max violation = {float(g_curr.max()):.2e} m")

    global_pose, joints = motion_data['global_pose'], motion_data['joints']
    # Build joints_new_ on DEVICE so all subsequent ops stay on-device.
    joints_new_ = target_joint_angles.clone()  # shape (T, J), already on DEVICE
    joints_new_[:, active_joint_ids] = joint_angles.detach()
    num_timesteps = joints_new_.shape[0]
    joints_new = torch.zeros((num_timesteps+INTERP_AMT+PAUSE_AMT, joints_new_.shape[1]), dtype=joints_new_.dtype, device=DEVICE)
    trans_new = torch.zeros((num_timesteps+INTERP_AMT+PAUSE_AMT, 3), dtype=joints_new_.dtype, device=DEVICE)
    quats_new = torch.zeros((num_timesteps+INTERP_AMT+PAUSE_AMT, 4), dtype=joints_new_.dtype, device=DEVICE)
    trans_new[INTERP_AMT + PAUSE_AMT:, :] = target_trans.clone()
    quats_new[INTERP_AMT + PAUSE_AMT:, :] = target_quats.clone()
    joints_new[INTERP_AMT + PAUSE_AMT:, :] = joints_new_
    quats_new[:INTERP_AMT + PAUSE_AMT, 0] = 1.0  # Set the first quaternion component to 1.0


    joints_new[:PAUSE_AMT, :] = torch.tensor(init_joint_angles, device=DEVICE).unsqueeze(0).repeat(PAUSE_AMT, 1)
    joints_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = joints_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
        (joints_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - joints_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
        torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)

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
            torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)
    else :
        trans_new[:PAUSE_AMT, 2] -= pos_left_ankle[:PAUSE_AMT, 2]
        trans_new[:PAUSE_AMT, :2] = pos_left_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_left_ankle[:PAUSE_AMT, :2]
        trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
            (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
            torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)

    # Interpolate quats from PAUSE_AMT to PAUSE_AMT + INTERP_AMT

    quats_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = slerp(
        quats_new[PAUSE_AMT-1:PAUSE_AMT, :],
        quats_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :],
        torch.linspace(0, 1, INTERP_AMT, device=DEVICE).unsqueeze(1)
    )

    Ts_world_root = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(quats_new.cpu())),jnp.array(trans_new.cpu())) 

    # Compute world-frame hand tip trajectory using the optimised joints_new.
    # Recompute rot_matrix from the fully-finalised quats_new (after slerp and ankle fix).
    rot_matrix_final = quaternion_to_matrix(quats_new)
    tf_hand = fk_results["right_rubber_hand"]
    hand_tf_mat = tf_hand.get_matrix()                        # (T, 4, 4)
    hand_pos = hand_tf_mat[:, :3, 3]                         # hand origin in root frame (T, 3)
    hand_rot = hand_tf_mat[:, :3, :3]                        # hand orientation in root frame (T, 3, 3)
    # Offset along hand's local x-axis (finger direction) in root frame, then to world.
    # This is the true geometric fingertip position and matches the collision cost in compute_cost.
    local_tip = torch.tensor([HAND_TIP_OFFSET, 0., 0.], device=DEVICE)
    tip_root = hand_pos + torch.bmm(
        hand_rot, local_tip.view(1, 3, 1).expand(hand_rot.shape[0], -1, -1)
    ).squeeze(-1)
    hand_tip_traj = torch.bmm(tip_root.unsqueeze(1), rot_matrix_final.transpose(2, 1))[:, 0] + trans_new
    hand_tip_traj = hand_tip_traj.detach().cpu().numpy()     # (T, 3)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    with open(SAVE_DIR + pkl_path[:-4] + "_n.pkl","wb") as f:
        # Move tensors to CPU before pickling so the saved file is device-agnostic.
        pickle.dump({"global_pose": Ts_world_root, "joints": joints_new.cpu(), "global_position": trans_new.cpu(),
                     "grab_pos": motion_data["grab_pos"], "grab_idx": motion_data["grab_idx"]+PAUSE_AMT+INTERP_AMT,
                     "hand_tip_traj": hand_tip_traj}, f)

if VISUALIZE:
    # Define cuboid dimensions (width, height, depth)
    width, height, depth = 1., OFFSET_Z, 2.0

    # Create a cuboid using Trimesh
    cuboid = trimesh.creation.box(extents=(width, depth, height))

    # Optionally, apply a transformation (e.g., move it to x=1, y=0.5, z=0)
    transform = np.eye(4)
    transform[:3, 3] = [grab_pos[0].item()+0.5+OFFSET_X, 0., OFFSET_Z/2.]
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
            urdf_vis_new.update_cfg(np.array(joints_new[tstep].cpu()))
