"""Adapter B: holosoma object_interaction output (.npz qpos F,43) -> TrajGen reference .pkl (29-DOF).

Runs in `dreamcontrol_51` (jaxlie + torch + pytorch_kinematics; NO mujoco).

holosoma qpos layout (F,43): [base_pos(3), base_quat(4 wxyz), 29 joints, object_pos(3), object_quat(4)].
The 29-joint block (qpos[7:36]) is ALREADY in the canonical TrajGen/Training 29-DOF order
(legs L, legs R, waist yaw/roll/pitch, arm L, arm R == motion_lib_base.py JointNamesOrder extended,
== g1_29dof.urdf revolute order), so the joint mapping is IDENTITY — no permutation.

Produces the .pkl schema consumed by Training/.../motion_lib_base.py:
  global_pose (jaxlie.SE3, batched F), joints (F,29 torch), global_position (F,3 torch),
  grab_pos (3, torch), grab_idx (int).
motion_lib uses global_pose.translation()/.rotation().wxyz, recomputes grab_pos from grab_idx+FK
(only the key's presence matters), and adds transl[:,2]+=0.035 -> so we ground the root per-frame
(lowest link -> 0), replicating sample/Pick_sim/retarget.py `global_position[:,2] -= min_z`.

Also re-inserts the 10-frame grab-hold (FREEZE_FOR, lift-delay) as in retarget.py, length-preserving.
NOTE: the old refine height-floor (max(z,offset_z)) is a reward-reference op, and holosoma's
object_interaction already enforces object/ground non-penetration -> omitted here (flag if needed).

Writes:
  <out>.pkl                    -> the reference (schema above)
  <out>_render.npz {qpos(F,43), grab_idx, grab_pos, fps}  -> grounded+held qpos for headless render
                                                             (render in the mujoco env separately)

Usage: python holosoma_to_pkl.py <holosoma_out.npz> <out_basepath>
"""
import numpy as np, torch, sys, os, pickle
import jaxlie, jax.numpy as jnp
import pytorch_kinematics as pk

FREEZE_FOR = 10
URDF = os.path.expanduser("~/kevin/DreamVLA/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf")

npz_path = sys.argv[1]
out_base = sys.argv[2]

D = np.load(npz_path)
q = np.array(D["qpos"]).astype(np.float64)                         # (F,43)
F = q.shape[0]
base_pos = q[:, 0:3].copy()
base_quat = q[:, 3:7].copy()                                       # wxyz
joints = q[:, 7:36].copy()                                         # (F,29) identity order
obj_pos = q[:, 36:39].copy()
obj_quat = q[:, 39:43].copy()


def quat_wxyz_to_R(quat):                                          # (F,4)->(F,3,3), same formula as render_skel2
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R = np.empty((quat.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# --- grab_idx: frame the hand reaches the object (last static object frame before the lift) ---
moved = np.linalg.norm(obj_pos - obj_pos[0][None], axis=-1)
grab_idx = int(max(0, np.argmax(moved > 1e-4) - 1))

# --- per-frame grounding: lowest link -> 0 (via FK), shift base + object by the same mz ---
chain = pk.build_chain_from_urdf(open(URDF, "rb").read())
fk = chain.forward_kinematics(torch.tensor(joints, dtype=torch.float32))      # dict link -> Transform3d (batched)
local = torch.stack([fk[l].get_matrix()[:, :3, 3] for l in fk], dim=1).numpy()  # (F, L, 3) in root frame
Rb = quat_wxyz_to_R(base_quat)                                                 # (F,3,3)
world_z = (np.einsum("fij,flj->fli", Rb, local) + base_pos[:, None, :])[:, :, 2]  # (F, L)
mz = world_z.min(axis=1)                                                       # (F,)
base_pos[:, 2] -= mz                                                           # robot: per-frame grounding (feet -> 0)
# Object grounding: the resting bottle is WORLD-FIXED pre-grab, so it must NOT inherit the
# robot's per-frame grounding bob (mz swings ~4cm during the reach). Use a CONSTANT shift while
# resting (through grab_idx), then track the per-frame shift once it's attached to the hand.
obj_shift = mz.copy()
obj_shift[:grab_idx + 1] = mz[grab_idx]
obj_pos[:, 2] -= obj_shift

# --- FREEZE_FOR hold at grab_idx (lift-delay), length-preserving (retarget.py 189-195) ---
def freeze(a):
    if 0 < grab_idx < F - FREEZE_FOR:
        a[grab_idx + FREEZE_FOR:] = a[grab_idx:F - FREEZE_FOR]
        a[grab_idx:grab_idx + FREEZE_FOR] = a[grab_idx]
    return a
base_pos = freeze(base_pos); base_quat = freeze(base_quat); joints = freeze(joints)
obj_pos = freeze(obj_pos); obj_quat = freeze(obj_quat)

# --- .pkl (schema consumed by motion_lib_base.py) ---
global_pose = jaxlie.SE3.from_rotation_and_translation(
    jaxlie.SO3(jnp.array(base_quat)), jnp.array(base_pos))
joints_t = torch.tensor(joints, dtype=torch.float32)
grab_pos = torch.tensor(obj_pos[grab_idx], dtype=torch.float32)                # hand world pos at grab (grounded)
pkl = {"global_pose": global_pose, "joints": joints_t,
       "global_position": torch.tensor(base_pos, dtype=torch.float32),
       "grab_pos": grab_pos, "grab_idx": grab_idx}
with open(out_base + ".pkl", "wb") as f:
    pickle.dump(pkl, f)

# --- companion render npz (grounded+held full qpos43; render in the mujoco env) ---
qpos_out = np.concatenate([base_pos, base_quat, joints, obj_pos, obj_quat], axis=1)
np.savez(out_base + "_render.npz", qpos=qpos_out, grab_idx=grab_idx,
         grab_pos=grab_pos.numpy(), fps=np.int64(30))

print(f"wrote {out_base}.pkl  F={F} joints={joints.shape[1]}DOF grab_idx={grab_idx} "
      f"ground_shift[min={mz.min():.3f},max={mz.max():.3f}]  grab_pos={np.round(grab_pos.numpy(),3)}")
print(f"wrote {out_base}_render.npz (qpos {qpos_out.shape}) for headless render")
