"""Adapter B: holosoma object_interaction output (.npz qpos F,43) -> TrajGen reference .pkl.

Runs in `dreamcontrol_51` (jaxlie + torch + pytorch_kinematics; NO mujoco).

DOF is selected by ``HS_PKL_DOF`` (default 29). Both variants are formatted to match the OLD
refine_motions_al.py output as closely as possible so the SONIC pipeline consumes them
identically (feed-in rate, joint order, start lead-in, grab hold, frozen left arm):
  * HS_PKL_DOF=29 (DEFAULT): keep the full 29-joint block -> waist roll/pitch TRACKED. The
    holosoma joint-block order [legs L6, legs R6, waist yaw/roll/pitch, arm L7, arm R7] already
    equals JointNamesOrder-29 (waist at 12/13/14), so no reordering -- just don't drop 13,14.
  * HS_PKL_DOF=27: drop the waist roll/pitch (block idx 13,14) -> the legacy validated 27-DOF
    reference [legs L6, legs R6, waist_yaw, arm L7, arm R7] (JointNamesOrder-27).

holosoma qpos (F,43): [base_pos(3), base_quat(4 wxyz), 29 joints, object_pos(3), object_quat(4)].

Pipeline (mirrors retarget.py + refine_motions_al.py):
  1. per-frame ground (lowest link-origin -> 0) via FK (g1_{27,29}dof.urdf).
  2. select DOF (drop waist roll/pitch for 27, keep all for 29).
  3. FREEZE_FOR=10 grab-hold at grab_idx (length-preserving), from object-motion onset.
  4. hard-freeze left arm across ALL frames (refine's "Make left arm non functional").
  5. PREPEND start lead-in: PAUSE(10) static at init pose + INTERP(10) ramp into the motion,
     root quat identity during PAUSE then slerp identity->motion[0] over INTERP, root trans held.
  6. grab_idx += PAUSE+INTERP.

Usage: python holosoma_to_pkl.py <holosoma_out.npz> <out_basepath>   -> writes <out>.pkl
       HS_PKL_DOF=27 python holosoma_to_pkl.py ...                    -> 27-DOF variant
"""
import numpy as np, torch, sys, os, pickle
import jaxlie, jax.numpy as jnp
import pytorch_kinematics as pk

PAUSE, INTERP, FREEZE_FOR = 10, 10, 10
DOF = int(os.environ.get("HS_PKL_DOF", "29"))
assert DOF in (27, 29), f"HS_PKL_DOF must be 27 or 29, got {DOF}"
_ROBOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "Training", "HumanoidVerse", "humanoidverse", "data", "robots", "g1")
URDF = os.path.join(_ROBOTS, f"g1_{DOF}dof.urdf")

# refine_motions_al.py init_joint_angles (frozen-left-arm pose). 27-DOF base; the 29-DOF
# variant inserts waist_roll=waist_pitch=0 after waist_yaw (block idx 12).
_INIT_LEGS = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0.]      # 12
_INIT_LARM = [0.35, 0.16, 0., 0.87, 0., 0., 0.]                                   # 7
_INIT_RARM = [0.35, -0.16, 0., 0.87, 0., 0., 0.]                                  # 7
if DOF == 27:
    INIT = np.array(_INIT_LEGS + [0.] + _INIT_LARM + _INIT_RARM, dtype=np.float64)          # waist_yaw only
    KEEP = [i for i in range(29) if i not in (13, 14)]   # drop waist roll/pitch
    LARM0 = 13                                            # left arm start idx (JointNamesOrder-27)
else:
    INIT = np.array(_INIT_LEGS + [0., 0., 0.] + _INIT_LARM + _INIT_RARM, dtype=np.float64)  # waist yaw/roll/pitch
    KEEP = list(range(29))                                # keep all 29
    LARM0 = 15                                            # left arm start idx (JointNamesOrder-29, +2 for waist roll/pitch)
assert INIT.shape[0] == DOF


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


def freeze_left_arm(j):   # j (F,DOF); left arm = idx LARM0..LARM0+6 (sh_pitch, sh_roll, sh_yaw, elbow, 3x wrist)
    j[:, LARM0:LARM0 + 7] = 0.0
    j[:, LARM0] = 0.35          # shoulder_pitch
    j[:, LARM0 + 1] = 0.16      # shoulder_roll
    j[:, LARM0 + 3] = 0.87      # elbow
    return j


npz_path, out_base = sys.argv[1], sys.argv[2]
q = np.array(np.load(npz_path)["qpos"]).astype(np.float64)          # (F,43)
F = q.shape[0]
base_pos = q[:, 0:3].copy(); base_quat = q[:, 3:7].copy()
joints = q[:, 7:36][:, KEEP].copy()                               # (F,DOF) JointNamesOrder
obj_pos = q[:, 36:39].copy()

# grab onset = last static-object frame before the lift
moved = np.linalg.norm(obj_pos - obj_pos[0][None], axis=1)
grab_idx = int(max(0, np.argmax(moved > 1e-4) - 1))

# --- per-frame grounding (lowest link-origin -> 0) via g1_{DOF}dof FK ---
chain = pk.build_chain_from_urdf(open(URDF, "rb").read())
fk = chain.forward_kinematics(torch.tensor(joints, dtype=torch.float32))
local = torch.stack([fk[l].get_matrix()[:, :3, 3] for l in fk], dim=1).numpy()   # (F,L,3)
mz = (np.einsum("fij,flj->fli", quat_wxyz_to_R(base_quat), local) + base_pos[:, None, :])[:, :, 2].min(axis=1)
base_pos[:, 2] -= mz

# --- freeze left arm across ALL frames (refine parity) ---
joints = freeze_left_arm(joints)

# --- FREEZE_FOR grab hold (length-preserving) ---
def freeze_hold(a):
    if 0 < grab_idx < F - FREEZE_FOR:
        a[grab_idx + FREEZE_FOR:] = a[grab_idx:F - FREEZE_FOR]
        a[grab_idx:grab_idx + FREEZE_FOR] = a[grab_idx]
    return a
base_pos = freeze_hold(base_pos); base_quat = freeze_hold(base_quat); joints = freeze_hold(joints)

# --- prepend PAUSE + INTERP start lead-in (refine parity) ---
a = np.linspace(0.0, 1.0, INTERP)[:, None]
j_lead = np.concatenate([np.tile(INIT, (PAUSE, 1)), INIT[None] * (1 - a) + joints[0][None] * a], axis=0)
joints = np.concatenate([j_lead, joints], axis=0)
tr_lead = np.tile(base_pos[0], (PAUSE + INTERP, 1))                 # hold root during lead-in
base_pos = np.concatenate([tr_lead, base_pos], axis=0)
q_lead = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (PAUSE + INTERP, 1))   # identity during PAUSE
q_lead[PAUSE:PAUSE + INTERP] = slerp(np.array([1.0, 0.0, 0.0, 0.0]), base_quat[0], np.linspace(0, 1, INTERP))
base_quat = np.concatenate([q_lead, base_quat], axis=0)
grab_idx += PAUSE + INTERP

# --- .pkl (DOF-DOF, OLD format) ---
global_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(base_quat)), jnp.array(base_pos))
grab_pos = torch.tensor(obj_pos[max(0, grab_idx - PAUSE - INTERP)], dtype=torch.float32)
pkl = {"global_pose": global_pose, "joints": torch.tensor(joints, dtype=torch.float32),
       "global_position": torch.tensor(base_pos, dtype=torch.float32),
       "grab_pos": grab_pos, "grab_idx": grab_idx}
with open(out_base + ".pkl", "wb") as f:
    pickle.dump(pkl, f)
_wr = np.linalg.norm(joints[:, 13]) if DOF == 29 else 0.0
_wp = np.linalg.norm(joints[:, 14]) if DOF == 29 else 0.0
print(f"wrote {out_base}.pkl  {DOF}-DOF  frames {F}->{joints.shape[0]} (lead-in {PAUSE}+{INTERP}) "
      f"grab_idx={grab_idx}  ground_shift[{mz.min():.3f},{mz.max():.3f}]  "
      f"waist_roll|pitch L2={_wr:.3f}|{_wp:.3f}  frame0 |dq/dt|@20fps="
      f"{np.linalg.norm((joints[1]-joints[0])/0.05):.3f} (want ~0)")
