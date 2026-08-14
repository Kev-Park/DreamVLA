"""29-DOF adaptation of refine_motions_al.py, restructured as an importable module.

The augmented-Lagrangian arm-refinement cost (compute_cost) is IDENTICAL to the legacy
refine_motions_al.py -- both g1_27dof and g1_29dof FK have 39 links with i==36 == right_wrist_
pitch_link and i==38 == right_rubber_hand, and compute_cost is joint-NAME-keyed -- so only the
joint_names / init / inactive lists change for 29-DOF. Exposes refine_arm(...) which runs the AL
loop on the RIGHT-ARM joints of the core motion and returns refined 29-DOF joints (no prepend --
Adapter B handles grounding / grab-hold / lead-in). Viz + file I/O + main loop removed.
"""
import os
import torch
import pytorch_kinematics as pk
import numpy as np
import torch.optim as optim
from isaac_utils.rotations import(
    quat_conjugate,
    quaternion_to_matrix,
    slerp
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load URDF and create kinematic chain (29-DOF; same 39 FK links as 27-DOF, i==36/38 identical)
urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'Training', 'HumanoidVerse', 'humanoidverse', 'data', 'robots', 'g1', 'g1_29dof.urdf')
chain = pk.build_chain_from_urdf(open(urdf_path, 'rb').read()).to(dtype=torch.float32, device=DEVICE)

# 29-DOF joint config (JointNamesOrder-29: waist_roll/pitch at 13,14; right arm 22-28 optimized).
JOINT_NAMES_29 = ['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint','waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']
INIT_29 = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0., 0., 0., 0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.]
INACTIVE_29 = [n for n in JOINT_NAMES_29 if not (n.startswith('right_shoulder') or n.startswith('right_elbow') or n.startswith('right_wrist'))]
SMOOTH_AMT = 20
PAUSE_AMT = 10
INTERP_AMT = 10
WRIST_TO_COLLISION = 0.35
VISUALIZE = False
OFFSET_Z = 0.86
OFFSET_X = -0.35
HAND_TIP_OFFSET = 0.15
# Forward-projection factor for the GRASP PALM target: palm = rubber_hand + HAND_FWD*(rubber_hand - wrist_yaw).
# The AL loop drives THIS palm point (not the wrist) to the object, so the palm — the grasp point the
# render draws as CYAN and the synthesized object ref tracks — lands on the object at grasp (no teleport).
# MUST match play_sonic_adapter.py --hand-fk-forward (default 1.5) for the render/reward to align.
HAND_FWD = 1.5
TIP_TO_TABLE_EDGE_TAPER_START_DIST = 0.05
TIP_SPEED_WINDOW = 50
SAVE_DIR = "../Pick_sim2/"
TRAJ_FPS_DEFAULT = 20.0
APPROACH_SPOOF_LEFT_X = 0.24
APPROACH_GATE_CLEARANCE = float(os.environ.get("HS_APPROACH_GATE_CLEARANCE", "0.15"))  # gate placed this far RIGHT of the object (m)
APPROACH_TAPER_LEAD = int(os.environ.get("HS_APPROACH_TAPER_LEAD", "8"))    # gate held until this many frames before grab
APPROACH_TAPER_WINDOW = int(os.environ.get("HS_APPROACH_TAPER_WINDOW", "20"))  # frames over which the gate releases (was hardcoded 10)
APPROACH_Z_CLEARANCE = float(os.environ.get("HS_APPROACH_Z_CLEARANCE", "-0.03"))  # z-gate: allowed height above object center (m)
DOWNVEL_W = float(os.environ.get("HS_DOWNVEL_W", "300.0"))                      # downward-velocity penalty weight (pre-grab)
GATE_END_SLACK = float(os.environ.get("HS_GATE_END_SLACK", "0.05"))              # gate line ends this far PAST the object (m)
LEVEL_LEAD = int(os.environ.get("HS_LEVEL_LEAD", "1000"))                    # level-hand term starts this many frames before grab
LEVEL_W = float(os.environ.get("HS_LEVEL_W", "150.0"))                      # level-hand orientation weight (soft)
# HARD levelness constraint (AL): (1 - up_z) <= LEVEL_HARD_EPS within the approach window
# [grab - LEVEL_HARD_LEAD, grab]. eps is 1 - cos(tilt): 0.03 ~= 14 deg max tilt. Enforced via the
# same dual/rho machinery as the table constraint, so a tilted high hover (m66: level fell to 0.45
# under the 150-weight soft term) becomes INFEASIBLE rather than merely expensive. Post-grab stays
# soft-only (the raw carry can be kinematically awkward; hard-constraining it can make AL diverge).
LEVEL_HARD_EPS = float(os.environ.get("HS_LEVEL_HARD_EPS", "0.03"))
LEVEL_HARD_LEAD = int(os.environ.get("HS_LEVEL_HARD_LEAD", "60"))
LEVEL_CONSTRAINT_TOL = 1e-3
POINT_W = float(os.environ.get("HS_POINT_W", "30.0"))                      # palm-pointing orientation weight (soft)
TORSO_CYLINDER_RADIUS = 0.125
WRIST_TORSO_SEGMENT_RADIUS = 0.03
HAND_TORSO_SEGMENT_RADIUS = 0.04
TORSO_CYLINDER_TOP_EXTENSION = 0.35
TORSO_COLLISIONS_ENABLED = False
HAND_APPROACH_SOFT_WEIGHT = float(os.environ.get("HS_HAND_APPROACH_W", "3000.0")) # rightward-gate bias

WRIST_GRAB_CHARB_WEIGHT = float(os.environ.get("HS_CHARB_W", "30.0")) # palm-to-object Charbonnier weight (final approach + hold)
# Charbonnier knee (m): pull is L1 (constant force) beyond eps, QUADRATIC (force ~ distance) inside.
# eps is the DECELERATION ZONE on arrival — 2e-3 gave a force cliff 2 mm from the target (constant-
# speed approach, dead stop = the m7 lunge-stop). 0.05 fades the force over the last 5 cm -> soft landing.
WRIST_GRAB_CHARB_EPS = float(os.environ.get("HS_CHARB_EPS", "0.05"))


def _smooth01(u):
    """C2 smootherstep 6u^5-15u^4+10u^3 on clamp(u,0,1): zero SLOPE at both ends. The previous
    clamp(u)^2 schedule was smooth at onset but had max slope AT the end-clamp -> the gate line
    (and charb weight) moved fastest at the instant it halted (te), a velocity discontinuity the
    hand inherited (m50 bump ~step 200 = te; m7 twitch at step 100 = te exactly)."""
    u = torch.clamp(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)

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
AL_RHO_GROWTH     = 3.0      # penalty multiplier each outer iteration (reduced for soft cost influence)
AL_RHO_INIT_TABLE = 10.0      # table-only initial penalty (slower start)
AL_RHO_GROWTH_TABLE = 1.5     # table-only penalty growth (even slower ramp)
AL_RHO_MAX        = 1e8       # cap so gradients don't explode for quadratic hard optimization
AL_OUTER_ITERS    = 40        # max AL outer iterations
AL_INNER_ITERS    = 300       # Adam steps per outer iteration

# Written by compute_cost every forward pass; read by the AL outer loop.
_last_g_t:         torch.Tensor | None = None
_last_g_cap_wrist: torch.Tensor | None = None
_last_g_dof_speed: torch.Tensor | None = None
_last_g_level:     torch.Tensor | None = None
_last_cost_terms: dict[str, torch.Tensor] | None = None
palm_target:       torch.Tensor | None = None   # world-frame object target the PALM is driven to (set by refine_arm)
palm_target_traj:  torch.Tensor | None = None   # (F,3) per-frame palm target: static grab point pre-grab, object trajectory post-grab

def compute_cost(joint_angles, trans, quats, offset_x=OFFSET_X, offset_z=OFFSET_Z, debug=False,
                 lambda_table: torch.Tensor | None = None,
                 lambda_wrist: torch.Tensor | None = None,
                 lambda_dof_speed: torch.Tensor | None = None,
                 lambda_level: torch.Tensor | None = None,
                 rho_al: float = AL_RHO_INIT,
                 rho_al_table: float | None = None):
    global _last_g_t, _last_g_cap_wrist, _last_g_dof_speed, _last_g_level, _last_cost_terms

    n_frames = joint_angles.shape[0]
    inv_n_frames = 1.0 / float(n_frames)
    wrist_dyn_contrib = torch.tensor(0.0, device=joint_angles.device, dtype=joint_angles.dtype)
    tip_dyn_contrib = torch.tensor(0.0, device=joint_angles.device, dtype=joint_angles.dtype)
    table_hard_contrib = torch.tensor(0.0, device=joint_angles.device, dtype=joint_angles.dtype)
    wrist_cap_hard_contrib = torch.tensor(0.0, device=joint_angles.device, dtype=joint_angles.dtype)
    approach_soft_contrib = torch.tensor(0.0, device=joint_angles.device, dtype=joint_angles.dtype)

    def al_penalty(g, lam_vec=None, rho_override: float | None = None):
        """Augmented-Lagrangian penalty for a hard constraint g >= 0.
        lam_vec: dedicated dual variable tensor; auto-sliced to len(g)."""
        rho_use = rho_al if rho_override is None else rho_override
        if lam_vec is not None:
            lam = lam_vec[:len(g)]
            return lam * g + (rho_use / 2.0) * g ** 2
        return (rho_use / 2.0) * g ** 2

    def segment_vertical_cylinder_collision_cost(p1_world, p2_world, cylinder_center_world,
                                                 z_low_world, z_high_world,
                                                 segment_thickness, cylinder_radius):
        """Finite arm segment capsule vs finite vertical torso cylinder.

        p1_world, p2_world: segment endpoints in world frame, shape (N, 3)
        cylinder_center_world: torso-cylinder centerline point, shape (N, 3)
        z_low_world, z_high_world: per-frame cylinder z bounds, shape (N,)
        Returns per-frame penetration depth >= 0, shape (N,)
        """
        r_total = segment_thickness + cylinder_radius
        sample_t = torch.linspace(0.0, 1.0, 7, device=p1_world.device, dtype=p1_world.dtype).view(1, -1, 1)
        pts = p1_world.unsqueeze(1) * (1.0 - sample_t) + p2_world.unsqueeze(1) * sample_t  # (N, 7, 3)

        center_xy = cylinder_center_world[:, :2].unsqueeze(1)  # (N, 1, 2)
        d_xy = torch.norm(pts[:, :, :2] - center_xy, dim=2)     # (N, 7)
        radial_pen = torch.relu(r_total - d_xy)                 # (N, 7)

        z_pts = pts[:, :, 2]                                     # (N, 7)
        z_low = z_low_world.unsqueeze(1)                         # (N, 1)
        z_high = z_high_world.unsqueeze(1)                       # (N, 1)
        z_gate_k = 60.0
        z_gate = torch.sigmoid(z_gate_k * (z_pts - z_low)) * torch.sigmoid(z_gate_k * (z_high - z_pts))

        pen = radial_pen * z_gate                                # (N, 7)
        return torch.max(pen, dim=1).values                      # (N,)

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
    trans_with_z_base = trans + torch.tensor([0., 0., 0.035], device=trans.device)

    torso_center_world = trans_with_z_base
    for _torso_link_name in ["waist_yaw_link", "torso_link", "base_link"]:
        if _torso_link_name in fk_results:
            torso_pos_root = fk_results[_torso_link_name].get_matrix()[:, :3, 3]
            torso_center_world = torch.bmm(torso_pos_root.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z_base
            break

    torso_top_world = torso_center_world
    for _upper_link_name in ["head_link", "neck_link", "chest_link", "upper_torso_link", "imu_link", "torso_link"]:
        if _upper_link_name in fk_results:
            upper_pos_root = fk_results[_upper_link_name].get_matrix()[:, :3, 3]
            torso_top_world = torch.bmm(upper_pos_root.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z_base
            break

    left_foot_world = torso_center_world
    right_foot_world = torso_center_world
    if "left_ankle_roll_link" in fk_results:
        left_pos_root = fk_results["left_ankle_roll_link"].get_matrix()[:, :3, 3]
        left_foot_world = torch.bmm(left_pos_root.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z_base
    if "right_ankle_roll_link" in fk_results:
        right_pos_root = fk_results["right_ankle_roll_link"].get_matrix()[:, :3, 3]
        right_foot_world = torch.bmm(right_pos_root.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z_base

    torso_z_low_world = torch.minimum(left_foot_world[:, 2], right_foot_world[:, 2])
    torso_z_high_world = torch.maximum(
        torso_top_world[:, 2] + TORSO_CYLINDER_TOP_EXTENSION,
        torso_z_low_world + 0.05,
    )

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
            wrist_speed_term = 50 * torch.abs(rel_dists) ** 2
            cost2[1:] += wrist_speed_term # L2 speed magnitude (dominates large speeds)
            wrist_dyn_contrib += torch.sum(wrist_speed_term) * inv_n_frames
            wrist_accel_term = 50 * (wrist_accel) ** 2
            cost2[1:-1] += wrist_accel_term  # acceleration
            wrist_dyn_contrib += torch.sum(wrist_accel_term) * inv_n_frames
            wrist_jerk = wrist_accel[1:] - wrist_accel[:-1]
            wrist_jerk_term = 50. * wrist_jerk ** 2
            cost2[2:-1] += wrist_jerk_term
            wrist_dyn_contrib += torch.sum(wrist_jerk_term) * inv_n_frames

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

            # [REF] Laziness: penalize deviation from rest pose, decaying toward grab_idx
            rest_pose = torch.tensor(init_joint_angles, device=DEVICE)[active_joint_ids]
            # Clamp effective end to actual sequence length — grab_idx can exceed N for some pkl files.
            laziness_end = min(max(grab_idx - 40, 0), joint_angles.shape[0])
            if laziness_end > 0:
                laziness_weight = torch.linspace(1.0, 0.0, laziness_end, device=DEVICE) ** 2
                cost2[:laziness_end] += laziness_weight * torch.sum(torch.abs(joint_angles[:laziness_end] - rest_pose), dim=1)

            # INTER-APPROACH COSTS

            # [ABS] Forearm capsule collision with grab obstacle — hard constraint via AL
            # The forearm capsule runs from the wrist origin (link 36) toward the hand origin (link 38).
            # We resolve the axis lazily inside compute_cost by storing wrist pos and computing
            # the direction to the hand link when i==38.  For the wrist pass we just stash the
            # transformed wrist position so the hand pass (i==38) can pick it up.
            _wrist_pos_world = transformed_keypts  # (N, 3) — used by i==38 for forearm axis

            # POST-APPROACH COSTS

            cost2[grab_idx:] += 10.*(joint_angles[grab_idx:, -6]+0.15)*(joint_angles[grab_idx:, -6]>-0.15)  # [ABS] Penalize right shoulder exceeding -0.15 (absolute joint limit)
            transformed_keypts_ref[grab_idx:, 2] = torch.maximum(transformed_keypts_ref[grab_idx:, 2], torch.tensor(offset_z, device=DEVICE)) # freeze reference height
            
            
            # Defer wrist-to-grab cost application to the hand-link block (i==38),
            # where the over-table trigger for distance-maximization is computed.
            _pending_wrist_grab_world = transformed_keypts


            # Post-grab raw-wrist tracking DEMOTED to height-only: the raw retarget's wrist path is
            # often inside-lying (left of the object) and was dragging the arm inside after grab.
            # Lateral position is now owned by the palm-on-object Charbonnier; keep only a weak z
            # guide so the arm doesn't sag.
            _pg_w = float(os.environ.get("HS_POSTGRAB_WRIST_W", "0.3"))
            cost2[grab_idx:] += _pg_w * torch.abs(transformed_keypts[grab_idx:, 2] - transformed_keypts_ref[grab_idx:, 2])

        elif i == 38: # right hand link (red dot in viser)
            rot_mat = tf.get_matrix()[:,:3,:3]
            rot_mat = torch.bmm(rot_matrix, rot_mat)

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

            # Forward-projected PALM (grasp point) in world frame, same formula as the render's CYAN:
            # palm = rubber_hand + HAND_FWD * (rubber_hand - wrist_yaw). Driven to the object below.
            _wy_root = fk_results["right_wrist_yaw_link"].get_matrix()[:, :3, 3]
            _wy_world = torch.bmm(_wy_root.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z
            palm_world = transformed_hand_orig + HAND_FWD * (transformed_hand_orig - _wy_world)

            # INTER-APPROACH COSTS

            # [ABS] Hard torso-vs-arm collision (AL, rho_other schedule).
            # Forearm segment runs from wrist origin (i==36) to hand origin (i==38), and
            # hand segment runs from hand origin to fingertip. Both must stay outside
            # the torso cylinder centered along robot spine.
            if '_wrist_pos_world' in locals():
                g_cap_forearm = segment_vertical_cylinder_collision_cost(
                    _wrist_pos_world,
                    transformed_hand_orig,
                    torso_center_world,
                    torso_z_low_world,
                    torso_z_high_world,
                    segment_thickness=WRIST_TORSO_SEGMENT_RADIUS,
                    cylinder_radius=TORSO_CYLINDER_RADIUS,
                )
                g_cap_hand = segment_vertical_cylinder_collision_cost(
                    transformed_hand_orig,
                    transformed_tip,
                    torso_center_world,
                    torso_z_low_world,
                    torso_z_high_world,
                    segment_thickness=HAND_TORSO_SEGMENT_RADIUS,
                    cylinder_radius=TORSO_CYLINDER_RADIUS,
                )
                g_cap_wrist = torch.maximum(g_cap_forearm, g_cap_hand)
            else:
                g_cap_wrist = torch.zeros(transformed_hand_orig.shape[0], device=DEVICE)
            if TORSO_COLLISIONS_ENABLED:
                _last_g_cap_wrist = g_cap_wrist.detach()
                wrist_cap_term = al_penalty(g_cap_wrist, lam_vec=lambda_wrist)
                cost2 += wrist_cap_term
                wrist_cap_hard_contrib += torch.sum(wrist_cap_term) * inv_n_frames
            else:
                _last_g_cap_wrist = torch.zeros_like(g_cap_wrist).detach()

            # Hand neutral orientation penalty: quadratic deviation from neutral straight hand pose
            # from grab_idx-20 onwards (palm vertical, pointing straight outward)
            # UNIFIED ORIENTATION objective -- LEVEL + POINTING, one window, applied together.
            #   LEVEL:    up-axis vertical (2 tilt DOF)  -> object held level
            #   POINTING: fingertip axis (local x) aimed horizontally at the object target (1 azimuth DOF)
            # Together they pin all 3 rotational DOF with a smooth target that rotates continuously as
            # the hand sweeps -- no azimuth slack for smoothing to fill, no onset transition. Direction
            # uses the HAND ORIGIN (stays ~0.2 m behind the palm target even at grasp -> never degenerate).
            orient_start = max(grab_idx - LEVEL_LEAD, 0)
            if orient_start < rot_mat.shape[0]:
                up_z = rot_mat[orient_start:, 2, 2]                    # world-z component of hand local z
                cost2[orient_start:] += LEVEL_W * (1.0 - up_z)         # = 1 - cos(tilt)
                _F = transformed_hand_orig.shape[0]
                _tg_o = (palm_target_traj[orient_start:_F]
                         if palm_target_traj is not None else palm_target.unsqueeze(0).expand(_F - orient_start, 3))
                _dvec = _tg_o - transformed_hand_orig[orient_start:]
                _dvec = torch.cat([_dvec[:, :2], torch.zeros_like(_dvec[:, 2:3])], dim=1)   # horizontal projection
                _pstar = _dvec / torch.norm(_dvec, dim=1, keepdim=True).clamp(min=1e-6)
                _xaxis = rot_mat[orient_start:, :, 0]                  # hand local x (fingertip axis) in world
                cost2[orient_start:] += POINT_W * (1.0 - (_xaxis * _pstar).sum(dim=1))

            # [ABS] HARD levelness (AL): (1 - up_z) <= LEVEL_HARD_EPS over the approach window
            # [grab - LEVEL_HARD_LEAD, grab]. Makes a tilted approach/hover INFEASIBLE (the soft
            # LEVEL_W term keeps gradient pressure toward perfectly flat inside the eps band).
            g_level_full = torch.zeros(rot_mat.shape[0], device=DEVICE, dtype=joint_angles.dtype)
            _hard_s = max(grab_idx - LEVEL_HARD_LEAD, 0)
            _hard_e = min(grab_idx + 1, rot_mat.shape[0])
            if _hard_e > _hard_s:
                g_level_full[_hard_s:_hard_e] = torch.relu(
                    (1.0 - rot_mat[_hard_s:_hard_e, 2, 2]) - LEVEL_HARD_EPS)
            _last_g_level = g_level_full.detach()
            cost2 += al_penalty(g_level_full, lam_vec=lambda_level)

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
            table_term = al_penalty(g_t, lam_vec=lambda_table, rho_override=rho_al_table)
            cost2[:table_gi] += table_term
            table_hard_contrib += torch.sum(table_term) * inv_n_frames


            # [ABS] Penalize hand-tip speed changes (acceleration), speed, and jerk
            tip_speed = torch.norm(transformed_tip[1:] - transformed_tip[:-1], dim=1)   # (N-1,) speed per frame
            tip_speed_term = 25 * torch.abs(tip_speed) ** 2
            cost2[1:] += tip_speed_term  # L2 speed magnitude (dominates large speeds)
            tip_dyn_contrib += torch.sum(tip_speed_term) * inv_n_frames
            tip_accel = tip_speed[1:] - tip_speed[:-1]
            tip_accel_term = 25 * (tip_accel) ** 2
            cost2[1:-1] += tip_accel_term  # acceleration
            tip_dyn_contrib += torch.sum(tip_accel_term) * inv_n_frames
            tip_jerk = tip_accel[1:] - tip_accel[:-1]
            tip_jerk_term = 50. * tip_jerk ** 2
            cost2[2:-1] += tip_jerk_term
            tip_dyn_contrib += torch.sum(tip_jerk_term) * inv_n_frames


            # [SOFT] Hand-tip approach shaping -- TWO-AXIS gate with a MOVING RELEASE LINE.
            # Sweep phase: gate lines sit a clearance RIGHT of the object (y) and AT grasp height (z),
            # full weight -> low right corridor. Release: instead of decaying the gate weight against
            # the charb (a 100:1 weight crossover whose equilibrium jumps ~93% through the window ->
            # park-then-rush), the gate LINES themselves travel to the object over the release window,
            # ending GATE_END_SLACK past it. The equilibrium (= line position) moves continuously, so
            # the hand walks in at line speed (~clearance/window), bounded by construction.
            _obj_y = palm_target[1] if palm_target is not None else capsule_obs_pos[1]
            _obj_z = palm_target[2] if palm_target is not None else capsule_obs_pos[2]
            _n_pre = grab_idx - 1
            taper_end = max(min(grab_idx - APPROACH_TAPER_LEAD, _n_pre), 1)
            taper_start = max(taper_end - APPROACH_TAPER_WINDOW, 1)
            frame_pre = torch.arange(1, grab_idx, device=DEVICE, dtype=joint_angles.dtype)
            _sline = _smooth01((frame_pre - float(taper_start)) / float(max(taper_end - taper_start, 1)))
            y_clear_t = (1.0 - _sline) * APPROACH_GATE_CLEARANCE + _sline * (-GATE_END_SLACK)
            z_clear_t = (1.0 - _sline) * APPROACH_Z_CLEARANCE + _sline * GATE_END_SLACK
            y_gate_t = _obj_y - y_clear_t                                   # moving Y line -> ends slack LEFT of obj
            z_gate_t = _obj_z + z_clear_t                                   # moving Z line -> ends slack ABOVE obj
            hand_y = transformed_tip[1:grab_idx, 1]                         # (G-1,) tip lateral pos
            hand_z = transformed_tip[1:grab_idx, 2]                         # (G-1,) tip height
            approach_pen = torch.relu(hand_y - y_gate_t) ** 2 + torch.relu(hand_z - z_gate_t) ** 2
            # DOWNWARD-VELOCITY penalty (untapered, pre-grab): early gradual descent cheapest.
            _dz = transformed_tip[1:grab_idx, 2] - transformed_tip[:grab_idx - 1, 2]
            cost2[1:grab_idx] += DOWNVEL_W * torch.relu(-_dz) ** 2

            n_trans = approach_pen.shape[0]
            if n_trans > 0:
                frame_ids = frame_pre
                distance_max_taper_start_frame = taper_start
                if palm_target is not None:
                    # Charb weight slaved to the SAME quadratic schedule as the moving gate line
                    # (_sline): attractor pressure grows exactly as fast as the line travels. The
                    # previous linear HALF-window ramp hit full weight while the line had only
                    # moved ~25% -> full attractor vs near-stationary 3000-gate -> the tip pierced
                    # the line ~3 cm and the wrist yawed (the m50/m7 dart-flick, probe frames
                    # 79-82 / 42-45). With matched schedules the pressure/barrier ratio is constant
                    # through the release, so the equilibrium rides the line with no transient.
                    palm_grab_start = max(distance_max_taper_start_frame + 1, 1)
                    palm_grab_end = palm_world.shape[0]
                    if palm_grab_end > palm_grab_start:
                        _fr = torch.arange(palm_grab_start, palm_grab_end, device=DEVICE, dtype=torch.float32)
                        _den = float(max(taper_end - taper_start, 1))
                        _s = _smooth01((_fr - float(taper_start)) / _den)   # same C2 schedule as the line
                        _tg = (palm_target_traj[palm_grab_start:palm_grab_end]
                               if palm_target_traj is not None else palm_target.unsqueeze(0))
                        palm_to_grab = palm_world[palm_grab_start:palm_grab_end] - _tg
                        palm_to_grab_norm = torch.norm(palm_to_grab, dim=1)
                        palm_grab_charb = _s * WRIST_GRAB_CHARB_WEIGHT * (
                            torch.sqrt(palm_to_grab_norm ** 2 + WRIST_GRAB_CHARB_EPS ** 2) - WRIST_GRAB_CHARB_EPS
                        )
                        cost2[palm_grab_start:palm_grab_end] += palm_grab_charb

                # Ramp in from trajectory start (unchanged). NO weight taper -- the LINE moves instead.
                ramp_end = max(min(grab_idx - 40, n_trans), 1)
                pre_ramp = torch.ones(n_trans, device=DEVICE, dtype=joint_angles.dtype)
                pre_mask = frame_ids <= float(ramp_end)
                pre_ramp[pre_mask] = (frame_ids[pre_mask] / float(ramp_end)) ** 2

                approach_term = HAND_APPROACH_SOFT_WEIGHT * pre_ramp * approach_pen
                cost2[1:grab_idx] += approach_term
                approach_soft_contrib += torch.sum(approach_term) * inv_n_frames
        else:
            continue

    mean_cost2 = torch.mean(cost2)
    total_cost = l2_cost + mean_cost2 + dof_speed_hard_cost
    _last_cost_terms = {
        "total": total_cost.detach(),
        "l2": l2_cost.detach(),
        "meanCost": mean_cost2.detach(),
        "dof_speed": dof_speed_hard_cost.detach(),
        "approach_soft": approach_soft_contrib.detach(),
        "table_hard": table_hard_contrib.detach(),
        "wrist_cap_hard": wrist_cap_hard_contrib.detach(),
        "tip_dyn": tip_dyn_contrib.detach(),
        "wrist_dyn": wrist_dyn_contrib.detach(),
    }
    return total_cost


def refine_arm(joints, base_pos, base_quat, grab_pos_obj, grab_idx_in, fps=20.0, verbose=False, obj_traj=None):
    """AL right-arm refinement on the CORE motion (before Adapter B's grab-hold/lead-in).

    joints (F,29) JointNamesOrder-29, base_pos (F,3), base_quat (F,4 wxyz), grab_pos_obj (3,)
    holosoma object xyz, grab_idx_in int. Returns refined joints (F,29) numpy with only the
    right-arm cols (22-28) changed. Runs on DEVICE (GPU strongly recommended: ~12k Adam steps).
    Logic is verbatim from refine_motions_al.py's per-motion setup + AL outer/inner loop.
    """
    global target_joint_angles, active_joint_names, inactive_joint_ids, joint_names
    global fk_results_ref, grab_idx, grab_pos, capsule_obs_pos, ref_dists, traj_fps_hz
    global active_joint_ids, init_joint_angles, palm_target, palm_target_traj

    joint_names = JOINT_NAMES_29
    init_joint_angles = INIT_29
    inactive_joint_names = INACTIVE_29
    active_joint_ids = [i for i, n in enumerate(joint_names) if n not in inactive_joint_names]
    active_joint_names = [n for n in joint_names if n not in inactive_joint_names]
    inactive_joint_ids = [i for i, n in enumerate(joint_names) if n in inactive_joint_names]

    target_joint_angles = torch.tensor(np.asarray(joints), dtype=torch.float32).to(DEVICE)
    trans = torch.tensor(np.asarray(base_pos), dtype=torch.float32).to(DEVICE)
    quats = torch.tensor(np.asarray(base_quat), dtype=torch.float32).to(DEVICE)
    grab_idx = int(grab_idx_in)
    traj_fps_hz = float(fps)

    q_dict = {name: target_joint_angles[:, i] for i, name in enumerate(joint_names)}
    fk_results_ref = chain.forward_kinematics(q_dict)
    pos = fk_results_ref["right_wrist_pitch_link"].get_matrix()[:, :3, 3]
    rot_matrix = quaternion_to_matrix(quats)
    trans_with_z_setup = trans + torch.tensor([0., 0., 0.035], device=DEVICE)
    wrist_keypts = torch.bmm(pos.unsqueeze(1), rot_matrix.transpose(2, 1))[:, 0] + trans_with_z_setup
    grab_pos = wrist_keypts[grab_idx].clone()
    grab_pos[0] += WRIST_TO_COLLISION                      # x_edge basis (table edge)
    # Wrist TARGET = the reference wrist's OWN grab position (grounded frame), where the
    # holosoma-retargeted hand actually grasps the object. NOT the raw grab_pos_obj: that is
    # ungrounded (z off by the per-frame grounding shift) and is the object CENTRE, ~0.1-0.15 m
    # forward of the wrist link -> driving the wrist to it over-reaches to a visibly-wrong target.
    # The reference already grasps the true (relocated) object, so its wrist point is object-correct.
    ref_dists = torch.norm(wrist_keypts[1:] - wrist_keypts[:-1], dim=1)

    # PALM target = the grounded holosoma object at grab (grab_pos_obj), in the FK world frame (+0.035
    # root lift, matching trans_with_z / motion_lib's object_poses). The AL loop drives the forward-
    # projected palm to THIS, so the palm (render CYAN, = the synthesized object ref post-grasp) lands
    # on the object.
    palm_target = torch.tensor(np.asarray(grab_pos_obj), dtype=torch.float32, device=DEVICE) \
        + torch.tensor([0., 0., 0.035], device=DEVICE)
    # Per-frame palm target: static grab point before grab; the MOVING object trajectory after
    # (so the full-window charb holds the palm ON the object through the lift/carry instead of
    # pinning the arm at the pickup point).
    if obj_traj is not None:
        _ot = torch.tensor(np.asarray(obj_traj), dtype=torch.float32, device=DEVICE) \
            + torch.tensor([0., 0., 0.035], device=DEVICE)
        palm_target_traj = palm_target.unsqueeze(0).repeat(target_joint_angles.shape[0], 1)
        _n = min(_ot.shape[0], palm_target_traj.shape[0])
        if grab_idx < _n:
            palm_target_traj[grab_idx:_n] = _ot[grab_idx:_n]
    else:
        palm_target_traj = None

    # Approach-avoid anchor = the HAND grasp point (the grounded object the palm is driven to), NOT the
    # reference WRIST. The 'swing away first, then taper in' shaping now repels the HAND from where it
    # will actually grasp, instead of from the wrist keypoint (which sat ~0.1-0.15 m short of the
    # object). Same spoof shift / taper / weight as before -- only the anchored point changes.
    capsule_obs_pos = palm_target.clone()

    joint_angles = torch.nn.Parameter(target_joint_angles[:, active_joint_ids].clone())

    # --- Augmented Lagrangian optimisation (outer: dual/rho update; inner: Adam) ---
    lambda_table = torch.zeros(joint_angles.shape[0], device=DEVICE)
    lambda_wrist = torch.zeros(joint_angles.shape[0], device=DEVICE)
    lambda_dof_speed = torch.zeros((joint_angles.shape[0] - 1) * joint_angles.shape[1], device=DEVICE)
    lambda_level = torch.zeros(joint_angles.shape[0], device=DEVICE)
    rho_al = AL_RHO_INIT
    rho_al_table = AL_RHO_INIT_TABLE
    converged = False
    g_curr = None
    for outer_iter in range(AL_OUTER_ITERS):
        optimizer = optim.Adam([joint_angles], lr=0.001)
        for _ in range(AL_INNER_ITERS):
            optimizer.zero_grad()
            cost = compute_cost(joint_angles, trans, quats,
                                lambda_table=lambda_table, lambda_wrist=lambda_wrist,
                                lambda_dof_speed=lambda_dof_speed, lambda_level=lambda_level,
                                rho_al=rho_al, rho_al_table=rho_al_table)
            cost.backward()
            optimizer.step()
        g_curr = _last_g_t
        g_curr_wrist = _last_g_cap_wrist
        g_curr_dof_speed = _last_g_dof_speed
        g_curr_level = _last_g_level
        if g_curr is None:
            break
        with torch.no_grad():
            lambda_table = torch.clamp(lambda_table + rho_al_table * g_curr, min=0.0)
            if TORSO_COLLISIONS_ENABLED and g_curr_wrist is not None:
                lambda_wrist = torch.clamp(lambda_wrist + rho_al * g_curr_wrist, min=0.0)
            if g_curr_dof_speed is not None:
                lambda_dof_speed = torch.clamp(lambda_dof_speed + rho_al * g_curr_dof_speed, min=0.0)
            if g_curr_level is not None:
                lambda_level = torch.clamp(lambda_level + rho_al * g_curr_level, min=0.0)
        max_viol = float(g_curr.max())
        max_viol_dof = float(g_curr_dof_speed.max()) if g_curr_dof_speed is not None else 0.0
        max_viol_level = float(g_curr_level.max()) if g_curr_level is not None else 0.0
        if verbose:
            print(f"[AL outer={outer_iter:02d}] table={max_viol:.2e}m dof_speed={max_viol_dof:.2e}rad/s "
                  f"level={max_viol_level:.2e} rho_table={rho_al_table:.1e} cost={float(cost):.4f}")
        if (max_viol < TABLE_CONSTRAINT_TOL and max_viol_dof < DOF_SPEED_CONSTRAINT_TOL
                and max_viol_level < LEVEL_CONSTRAINT_TOL):
            converged = True
            break
        rho_al_table = min(rho_al_table * AL_RHO_GROWTH_TABLE, AL_RHO_MAX)
        rho_al = min(rho_al * AL_RHO_GROWTH, AL_RHO_MAX)

    out = target_joint_angles.clone()
    out[:, active_joint_ids] = joint_angles.detach()
    max_move = float((out[:, active_joint_ids] - target_joint_angles[:, active_joint_ids]).abs().max())
    fv = float(g_curr.max()) if g_curr is not None else -1.0
    fvl = float(_last_g_level.max()) if _last_g_level is not None else -1.0
    print(f"[refine-al] grab_idx={grab_idx} AL {'converged' if converged else 'maxiter'} "
          f"final_table_viol={fv:.2e}m final_level_viol={fvl:.2e} max_arm_move={max_move:.3f}rad")
    return out.detach().cpu().numpy()
