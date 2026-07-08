"""Adapter B: holosoma object_interaction output (.npz qpos F,43) -> TrajGen reference .pkl.

Runs in `dreamcontrol_51` (jaxlie + torch + pytorch_kinematics; NO mujoco).

DEFAULT = 27-DOF, formatted to match the OLD refine_motions_al.py output as closely as possible
so the VALIDATED 27-DOF SONIC pipeline consumes it identically (feed-in rate, joint order, start
lead-in, grab hold, frozen left arm). Waist roll/pitch are DROPPED here (the 29-DOF upgrade is
re-added later once 27-DOF tracks cleanly). Set HS_PKL_DOF=29 for the 29-DOF variant (TODO).

holosoma qpos (F,43): [base_pos(3), base_quat(4 wxyz), 29 joints, object_pos(3), object_quat(4)].
The 29-joint block is [legs L6, legs R6, waist yaw/roll/pitch, arm L7, arm R7]; dropping the waist
roll/pitch (block idx 13,14) yields the OLD JointNamesOrder-27 [legs L6, legs R6, waist_yaw, arm L7, arm R7].

Pipeline (mirrors retarget.py + refine_motions_al.py):
  1. per-frame ground (lowest link-origin -> 0) via FK (g1_27dof.urdf, 27-DOF joints).
  2. drop waist roll/pitch -> 27-DOF joints.
  3. FREEZE_FOR=10 grab-hold at grab_idx (length-preserving), from object-motion onset.
  4. hard-freeze left arm across ALL frames: joints[:,13:20]=0; [13]=0.35,[14]=0.16,[16]=0.87
     (refine's "Make left arm non functional").
  5. PREPEND start lead-in: PAUSE(10) static at init pose + INTERP(10) ramp into the motion,
     root quat identity during PAUSE then slerp identity->motion[0] over INTERP, root trans held.
  6. grab_idx += PAUSE+INTERP.

Usage: python holosoma_to_pkl.py <holosoma_out.npz> <out_basepath>   -> writes <out>.pkl
"""
import numpy as np, torch, sys, os, pickle
import jaxlie, jax.numpy as jnp
import pytorch_kinematics as pk

PAUSE, INTERP, FREEZE_FOR = 10, 10, 10
URDF27 = os.path.expanduser("~/kevin/DreamVLA/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf")
# refine_motions_al.py init_joint_angles (27-DOF, includes the frozen-left-arm pose)
INIT27 = np.array([-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0.,
                   0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.], dtype=np.float64)
KEEP27 = [i for i in range(29) if i not in (13, 14)]   # drop waist roll/pitch from the 29-joint block


def quat_wxyz_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def slerp(q0, q1, ts):
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        out = q0[None] * (1 - ts[:, None]) + q1[None] * ts[:, None]
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    th = np.arccos(d); s = np.sin(th)
    return (np.sin((1 - ts) * th)[:, None] * q0[None] + np.sin(ts * th)[:, None] * q1[None]) / s


def freeze_left_arm27(j):   # j (F,27), JointNamesOrder-27 (left arm = idx 13..19)
    j[:, 13:20] = 0.0; j[:, 13] = 0.35; j[:, 14] = 0.16; j[:, 16] = 0.87
    return j


npz_path, out_base = sys.argv[1], sys.argv[2]
q = np.array(np.load(npz_path)["qpos"]).astype(np.float64)          # (F,43)
F = q.shape[0]
base_pos = q[:, 0:3].copy(); base_quat = q[:, 3:7].copy()
joints27 = q[:, 7:36][:, KEEP27].copy()                            # (F,27) OLD JointNamesOrder
obj_pos = q[:, 36:39].copy()

# grab onset = last static-object frame before the lift
moved = np.linalg.norm(obj_pos - obj_pos[0][None], axis=1)
grab_idx = int(max(0, np.argmax(moved > 1e-4) - 1))

# --- per-frame grounding (lowest link-origin -> 0) via g1_27dof FK ---
chain = pk.build_chain_from_urdf(open(URDF27, "rb").read())
fk = chain.forward_kinematics(torch.tensor(joints27, dtype=torch.float32))
local = torch.stack([fk[l].get_matrix()[:, :3, 3] for l in fk], dim=1).numpy()   # (F,L,3)
mz = (np.einsum("fij,flj->fli", quat_wxyz_to_R(base_quat), local) + base_pos[:, None, :])[:, :, 2].min(axis=1)
base_pos[:, 2] -= mz

# --- freeze left arm across ALL frames (refine parity) ---
joints27 = freeze_left_arm27(joints27)

# --- FREEZE_FOR grab hold (length-preserving) ---
def freeze_hold(a):
    if 0 < grab_idx < F - FREEZE_FOR:
        a[grab_idx + FREEZE_FOR:] = a[grab_idx:F - FREEZE_FOR]
        a[grab_idx:grab_idx + FREEZE_FOR] = a[grab_idx]
    return a
base_pos = freeze_hold(base_pos); base_quat = freeze_hold(base_quat); joints27 = freeze_hold(joints27)

# --- prepend PAUSE + INTERP start lead-in (refine parity) ---
a = np.linspace(0.0, 1.0, INTERP)[:, None]
j_lead = np.concatenate([np.tile(INIT27, (PAUSE, 1)), INIT27[None] * (1 - a) + joints27[0][None] * a], axis=0)
joints27 = np.concatenate([j_lead, joints27], axis=0)
tr_lead = np.tile(base_pos[0], (PAUSE + INTERP, 1))                 # hold root during lead-in
base_pos = np.concatenate([tr_lead, base_pos], axis=0)
q_lead = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (PAUSE + INTERP, 1))   # identity during PAUSE
q_lead[PAUSE:PAUSE + INTERP] = slerp(np.array([1.0, 0.0, 0.0, 0.0]), base_quat[0], np.linspace(0, 1, INTERP))
base_quat = np.concatenate([q_lead, base_quat], axis=0)
grab_idx += PAUSE + INTERP

# --- .pkl (27-DOF, OLD format) ---
global_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(base_quat)), jnp.array(base_pos))
grab_pos = torch.tensor(obj_pos[max(0, grab_idx - PAUSE - INTERP)], dtype=torch.float32)
pkl = {"global_pose": global_pose, "joints": torch.tensor(joints27, dtype=torch.float32),
       "global_position": torch.tensor(base_pos, dtype=torch.float32),
       "grab_pos": grab_pos, "grab_idx": grab_idx}
with open(out_base + ".pkl", "wb") as f:
    pickle.dump(pkl, f)
print(f"wrote {out_base}.pkl  27-DOF  frames {F}->{joints27.shape[0]} (lead-in {PAUSE}+{INTERP}) "
      f"grab_idx={grab_idx}  ground_shift[{mz.min():.3f},{mz.max():.3f}]  frame0 |dq/dt|@20fps="
      f"{np.linalg.norm((joints27[1]-joints27[0])/0.05):.3f} (want ~0)")
