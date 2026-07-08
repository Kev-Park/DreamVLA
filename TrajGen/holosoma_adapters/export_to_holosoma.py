"""Adapter A: OmniControl results.npy (SMPL 22-joint kpts) -> holosoma smplx .npz input.

OmniControl's 22 joints == holosoma SMPLX_DEMO_JOINTS order (same joints, same order), so the
output feeds holosoma's existing `--data-format smplx` directly (no custom format needed).
rot_mat replicates sample/*/retarget.py (Y-up -> Z-up).

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

R_WRIST = 21  # index of R_Wrist in SMPLX_DEMO_JOINTS (22-joint order)

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

# --- fabricate object_poses from the right wrist ---
# The object rests at the grab location until the hand reaches it, then rigidly tracks the wrist
# (the lift). To co-locate the object with the *retargeted* right hand we must pre-invert
# holosoma.preprocess_motion_data, which transforms human joints and object_poses ASYMMETRICALLY:
#   human joints: (z -= z_min) then *scale   (all axes scaled, z shifted to put feet on ground)
#   object_poses: out_xy = scale*in_xy;  out_z = in_z0 + scale*(in_z - in_z0)  (z0 NOT scaled)
# We choose the object INPUT so its OUTPUT equals the transformed right hand at every frame.
ROBOT_HEIGHT_G1 = 1.32          # holosoma config_types/robot.py g1 robot_height; smpl_scale = ROBOT_HEIGHT/height
L_FOOT, R_FOOT = 10, 11         # SMPLX_DEMO_JOINTS foot indices (holosoma toe_names for smplx)

wrist = gj[:, R_WRIST, :].astype(np.float64)                       # (N, 3), raw human frame
grab_t = int(np.argmin(wrist[:, 2]))                               # reach-down grab = lowest wrist
obj = wrist.copy()
obj[:grab_t] = wrist[grab_t]                                       # static at grab loc until reached

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
