"""Filter holosoma pick references by the LEGACY retarget.py trajectory-quality guards.

The old PyRoki retarget.py (sample/Pick_sim/retarget.py) SKIPPED motions failing any of four
criteria, so the yield was < 100. The holosoma pipeline dropped all of them and keeps every
motion, including ones the old pipeline would discard (esp. the "moving forward after grabbing"
overshoot -> the robot walks its root into the table). This re-implements those guards with the
SAME thresholds and emits the passing subset.

Guards (identical thresholds to retarget.py lines 178-212):
  1. min_dist > 0.15   -- source right-hand closest approach to the pick hint (first 140 frames)
  2. min_y   < 0.70    -- source right-hand minimum height (SMPL is Y-up)
  3. facing  > 45 deg  -- retargeted root max angle of its x-axis off world +x
  4. overshoot: max(root_x) + 0.15 > grab_x  -- root walks to/past the grab point

Guards 1-2 use the OmniControl source (results.npy, same file Adapter A reads); 3-4 use the
retargeted .pkl. .pkl id == results.npy motion index (Adapter A processes each id 1:1).

Usage: python filter_motions.py <pkl_dir> [results.npy] [out_dir]
  <pkl_dir>     dir of pick_<id>.pkl to filter
  results.npy   default ../sample/Pick_sim/results.npy (same as export_to_holosoma.py)
  out_dir       default <pkl_dir>_filtered ; passing pkls are symlinked in
"""
import os, sys, glob, pickle
import numpy as np

RIGHT_HAND_INDEX = 21
MIN_DIST_MAX = 0.15
MIN_Y_MIN = 0.70
FACE_DEG_MAX = 45.0
OVERSHOOT_MARGIN = 0.15

pkl_dir = sys.argv[1]
results_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "sample", "Pick_sim", "results.npy")
out_dir = sys.argv[3] if len(sys.argv) > 3 else pkl_dir.rstrip("/") + "_filtered"

DATA = np.load(results_path, allow_pickle=True).item()
hints = DATA["hint"][:, 1:]                                   # (k, H, 22, 3)
smpl_keypoints_ = DATA["motion"].transpose((0, 3, 1, 2))     # (k, N, 22, 3)


def R_from_wxyz(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xax = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)  # (F,3)
    return xax


def source_metrics(i):
    """min_dist (hand->pick hint, first 140 frames) and min_y (hand height), SOURCE space."""
    hint = hints[i]
    right_hand_pos = np.zeros(3)
    for joint_pos in hint:                                   # first hint frame with a valid pick
        rhp = joint_pos[RIGHT_HAND_INDEX]
        if rhp[2] > 0.0:
            right_hand_pos = rhp
            break
    dists = np.linalg.norm(smpl_keypoints_[i, :140, RIGHT_HAND_INDEX] - right_hand_pos[None, :], axis=-1)
    return float(dists.min()), float(np.min(smpl_keypoints_[i, :, RIGHT_HAND_INDEX, 1]))


os.makedirs(out_dir, exist_ok=True)
paths = sorted(glob.glob(os.path.join(pkl_dir, "pick_*.pkl")), key=lambda p: int(os.path.basename(p)[5:-4]))
passed, failed = [], []
print(f"filtering {len(paths)} refs from {pkl_dir}")
print(f"{'id':>4} {'min_dist':>8} {'min_y':>6} {'face':>6} {'over':>6}  verdict")
for p in paths:
    i = int(os.path.basename(p)[5:-4])
    d = pickle.load(open(p, "rb"))
    gp = np.asarray(d["global_position"]); grab = np.asarray(d["grab_pos"])
    quat = np.asarray(d["global_pose"].rotation().wxyz)
    md, my = source_metrics(i)
    face_deg = float(np.degrees(np.arccos(np.clip(R_from_wxyz(quat)[:, 0], -0.99, 0.99))).max())
    over = float(gp[:, 0].max() + OVERSHOOT_MARGIN - grab[0])   # >0 => fail
    reasons = []
    if md > MIN_DIST_MAX: reasons.append("dist")
    if my < MIN_Y_MIN: reasons.append("lowhand")
    if face_deg > FACE_DEG_MAX: reasons.append("facing")
    if over > 0: reasons.append("overshoot")
    ok = not reasons
    (passed if ok else failed).append(i)
    print(f"{i:>4} {md:>8.3f} {my:>6.2f} {face_deg:>6.1f} {over:>+6.2f}  {'PASS' if ok else 'DROP:'+','.join(reasons)}")
    if ok:
        dst = os.path.join(out_dir, f"pick_{i}.pkl")
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.abspath(p), dst)

print(f"\nYIELD: {len(passed)}/{len(paths)} passed  ->  {out_dir}")
print(f"passed ids: {passed}")
print(f"dropped {len(failed)}: {failed}")
