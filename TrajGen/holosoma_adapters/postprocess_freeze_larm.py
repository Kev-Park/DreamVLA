"""Post-process: freeze the G1 left arm in a holosoma object_interaction output (.npz qpos F,43).

Only the right arm is used for grasps, so the reference pins the left arm to a fixed natural pose.
This replicates sample/Pick_sim1/refine_motions_al.py "# Make left arm non functional" (a joint-space
overwrite applied AFTER retargeting), mapped from the old 27-DOF order to the 29-DOF qpos here.

29-DOF qpos left-arm indices (base = qpos[0:7]):
  22 left_shoulder_pitch, 23 left_shoulder_roll, 24 left_shoulder_yaw,
  25 left_elbow, 26 left_wrist_roll, 27 left_wrist_pitch, 28 left_wrist_yaw

Values (rad) from refine: shoulder_pitch=0.35, shoulder_roll=0.16, elbow=0.87, rest=0.

Usage: python postprocess_freeze_larm.py <in.npz> [out.npz]   (default out = <in>_frozen.npz)
This will fold into Adapter B (holosoma -> .pkl); kept standalone for the CKPT1b render.
"""
import numpy as np, sys

FREEZE = {22: 0.35, 23: 0.16, 24: 0.0, 25: 0.87, 26: 0.0, 27: 0.0, 28: 0.0}

inp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".npz", "_frozen.npz")
d = dict(np.load(inp, allow_pickle=True))
q = np.array(d["qpos"]).copy()
for i, v in FREEZE.items():
    q[:, i] = v
d["qpos"] = q
np.savez(out, **d)
print(f"wrote {out}  froze left-arm qpos[{min(FREEZE)}:{max(FREEZE)+1}] -> "
      f"sh_pitch=0.35 sh_roll=0.16 elbow=0.87 (rest 0)")
