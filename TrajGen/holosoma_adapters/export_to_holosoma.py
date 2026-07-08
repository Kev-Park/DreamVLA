"""Adapter A: OmniControl results.npy (SMPL 22-joint kpts) -> holosoma smplx .npz input.

OmniControl's 22 joints == holosoma SMPLX_DEMO_JOINTS order (same joints, same order), so the
output feeds holosoma's existing `--data-format smplx` directly (no custom format needed).
rot_mat replicates sample/*/retarget.py (Y-up -> Z-up).

Left arm is FROZEN on input (only the right arm grasps): the L_Shoulder/L_Elbow/L_Wrist
keypoints are held rigid relative to the torso frame (at the most rest-like reference frame),
so holosoma solves a BALANCED whole-body pose with a still left arm (vs. a post-hoc joint
overwrite that would unbalance the body). Replaces the joint-space freeze in
sample/Pick_sim1/refine_motions_al.py ("Make left arm non functional").

Also fabricates `object_poses` (T,7) [qw,qx,qy,qz,x,y,z] for object_interaction mode:
the grasped object (mustard bottle) rests at the grab location until the hand reaches it,
then is co-located with / tracks the right wrist through the lift. Orientation is identity
(the mustard.obj is baked Z-up upright). Positions are in the raw (pre-scale) human frame,
which is what holosoma's preprocess_motion_data expects. holosoma ignores object_poses in
robot_only mode, so writing it is harmless there.

Usage: python export_to_holosoma.py <motion_idx> [results.npy] [out_dir]
  -> writes <out_dir>/pick_<motion_idx>.npz with {global_joint_positions (N,22,3), height, object_poses (N,7)}
"""
import numpy as np, sys, os

# SMPLX_DEMO_JOINTS (22-joint order) indices
PELVIS, L_HIP, R_HIP, NECK = 0, 1, 2, 12
L_SHOULDER, L_ELBOW, L_WRIST = 16, 18, 20
R_WRIST = 21
FREEZE_LEFT_ARM = True
ROBOT_HEIGHT_G1 = 1.32   # holosoma config_types/robot.py g1 robot_height (smpl_scale = ROBOT_HEIGHT/height)


def _torso_frame(p, neck, lhip, rhip):
    """Right-handed torso frame (cols = world axes): up = pelvis->neck, lateral = Lhip->Rhip."""
    u = neck - p; u = u / (np.linalg.norm(u) + 1e-9)
    rr = rhip - lhip; rr = rr - (rr @ u) * u; rr = rr / (np.linalg.norm(rr) + 1e-9)
    f = np.cross(u, rr)
    return np.stack([rr, f, u], axis=1)


def freeze_left_arm(gj):
    """Hold L_Shoulder/L_Elbow/L_Wrist rigid w.r.t. the torso frame at the most rest-like frame
    (lowest left wrist), so the left arm moves rigidly with the torso -> still, balanced."""
    g = gj.astype(np.float64)
    ref = int(np.argmin(g[:, L_WRIST, 2]))
    Rref = _torso_frame(g[ref, PELVIS], g[ref, NECK], g[ref, L_HIP], g[ref, R_HIP])
    offs = {j: Rref.T @ (g[ref, j] - g[ref, PELVIS]) for j in (L_SHOULDER, L_ELBOW, L_WRIST)}
    out = g.copy()
    for t in range(g.shape[0]):
        Rt = _torso_frame(g[t, PELVIS], g[t, NECK], g[t, L_HIP], g[t, R_HIP])
        for j, off in offs.items():
            out[t, j] = g[t, PELVIS] + Rt @ off
    return out.astype(gj.dtype)

motion_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 20
results = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/kevin/DreamVLA/TrajGen/sample/Pick_sim/results.npy")
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/kevin/hs_input")
os.makedirs(out_dir, exist_ok=True)

DATA = np.load(results, allow_pickle=True).item()
smpl = DATA["motion"].transpose((0, 3, 1, 2))                       # (k, N, 22, 3)
rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
smpl = np.einsum("ij,ankj->anki", rot, smpl.copy())                 # Y-up -> Z-up
smpl = smpl[:, 1:]                                                  # drop frame 0 (match retarget.py)
gj = smpl[motion_idx].astype(np.float32)                           # (N, 22, 3)

h = float(gj[..., 2].max() - gj[..., 2].min())                     # rough standing height
if not (1.3 < h < 2.2):
    h = 1.78

if FREEZE_LEFT_ARM:
    gj = freeze_left_arm(gj)                                        # still left arm -> balanced solve

# --- fabricate object_poses from the right wrist ---
# The object rests at the grab location until the hand reaches it, then rigidly tracks the wrist
# (the lift). To co-locate the object with the *retargeted* right hand we must pre-invert
# holosoma.preprocess_motion_data, which transforms human joints and object_poses ASYMMETRICALLY:
#   human joints: (z -= z_min) then *scale   (all axes scaled, z shifted to put feet on ground)
#   object_poses: out_xy = scale*in_xy;  out_z = in_z0 + scale*(in_z - in_z0)  (z0 NOT scaled)
# We choose the object INPUT so its OUTPUT equals the transformed right hand at every frame.
L_FOOT, R_FOOT = 10, 11         # SMPLX_DEMO_JOINTS foot indices (holosoma toe_names for smplx)

wrist = gj[:, R_WRIST, :].astype(np.float64)                       # (N, 3), rotated Z-up frame
pelvis = gj[:, PELVIS, :].astype(np.float64)
# Grab frame = MAX REACH TOWARD THE OBJECT (arm fully extended to pick up) — a semantically
# equivalent point across trajectories. reach = horizontal projection of (wrist - pelvis) onto
# the pelvis->pick_point direction; grab = its argmax over the first 140 frames. The object then
# rests at the reach point, so the hand grabs at full extension rather than overshooting/returning.
# (pick_point = first OmniControl hint frame with right-hand z>0, per sample/Pick_sim/retarget.py.)
hint = DATA["hint"][:, 1:][motion_idx]                             # (N, 22, 3), raw Y-up (drop frame0)
pick_point_raw = next((np.asarray(jp[R_WRIST], np.float64) for jp in hint if jp[R_WRIST][2] > 0.0), None)
horizon = min(140, wrist.shape[0])
if pick_point_raw is not None:
    pp = rot @ pick_point_raw                                     # raw -> Z-up frame
    d = pp[None, :2] - pelvis[:, :2]                              # per-frame horizontal dir to object
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
    reach = ((wrist[:, :2] - pelvis[:, :2]) * d).sum(-1)          # forward reach toward object
else:
    reach = np.linalg.norm(wrist[:, :2] - pelvis[:, :2], axis=-1)  # fallback: horizontal reach
# Lift-aware grab selection: the object rests at wrist[grab] then tracks the wrist, so a valid grasp
# needs the wrist to RISE after grab (else no lift -> "not held"). Grab = MAX reach toward the object
# among only the frames that are FOLLOWED BY A LIFT. This keeps the natural full-extension pick (motion
# 20 @46) yet rejects late reach re-extensions with no lift after them (motion 25's argmax @124 -> the
# real lifting pick @~48). Threshold is in the human frame but targets a >=0.06 OUTPUT lift (holosoma
# scales the robot by ~height/1.32). If no frame has a lift, fall back to the frame closest to the object.
lift_min_human = 0.06 * h / ROBOT_HEIGHT_G1
lift_after = lambda g: float(wrist[g:, 2].max() - wrist[g, 2])
lift_ok = np.array([lift_after(g) >= lift_min_human for g in range(horizon)])
if lift_ok.any():
    grab_t = int(np.argmax(np.where(lift_ok, reach[:horizon], -np.inf)))
elif pick_point_raw is not None:
    grab_t = int(np.argmin(np.linalg.norm(wrist - pp[None, :], axis=1)))
else:
    grab_t = int(np.argmax(reach[:horizon]))
obj = wrist.copy()
obj[:grab_t] = wrist[grab_t]                                       # static at reach point until grabbed

scale = ROBOT_HEIGHT_G1 / h
z_min = float(gj[:, [L_FOOT, R_FOOT], 2].min())
if z_min >= 0.1:                                                   # matches preprocess mat_height branch
    z_min -= 0.1
A = obj[0, 2] - z_min                                             # grab height above ground (raw human frame)

N = gj.shape[0]
object_poses = np.zeros((N, 7), dtype=np.float64)
object_poses[:, 0] = 1.0                                           # identity quat (mustard.obj baked upright Z-up)
object_poses[:, 4:6] = obj[:, :2]                                 # preprocess scales x,y -> matches scaled hand
object_poses[:, 6] = scale * A + (obj[:, 2] - obj[0, 2])          # pre-inverted z (preprocess re-scales deltas)

np.savez(os.path.join(out_dir, f"pick_{motion_idx}.npz"),
         global_joint_positions=gj, height=np.float32(h), object_poses=object_poses)
print(f"wrote pick_{motion_idx}.npz  frames={N} joints={gj.shape[1]} "
      f"height={h:.2f}  z[{gj[...,2].min():.2f},{gj[...,2].max():.2f}]  "
      f"grab_t={grab_t}  scale={scale:.3f}  obj_out_z0={scale*A:.3f}  obj_grab_xy={np.round(obj[grab_t,:2],3)}")
