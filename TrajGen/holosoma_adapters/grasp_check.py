"""Reference-level grasp check on a holosoma object_interaction output (.npz qpos F,43).
HELD iff: grab detected (object moved), object LIFTED post-grab (>0.05m), and object CO-LOCATED
with the right hand at the final frame (<0.15m). Prints one CSV row: id,status,held,grab,moved,lift,dist."""
import numpy as np, mujoco, sys
npz, xml, mid = sys.argv[1], sys.argv[2], sys.argv[3]
q = np.load(npz)["qpos"]; obj = q[:, 36:39]
moved = np.linalg.norm(obj - obj[0][None], axis=1)
has = bool((moved > 1e-4).any()); grab = int(np.argmax(moved > 1e-4) - 1) if has else -1
mmax = float(moved.max())
m = mujoco.MjModel.from_xml_path(xml); dd = mujoco.MjData(m); hid = m.body("right_rubber_hand_link").id
e = q.shape[0] - 1; dd.qpos[:] = q[e]; mujoco.mj_forward(m, dd); h = dd.xpos[hid].copy()
dist = float(np.linalg.norm(h - obj[e]))
lift = float(obj[grab:, 2].max() - obj[grab, 2]) if grab >= 0 else 0.0
held = int((mmax > 0.05) and (lift > 0.05) and (dist < 0.15))
print(f"{mid},ok,{held},{grab},{mmax:.3f},{lift:.3f},{dist:.3f}")
