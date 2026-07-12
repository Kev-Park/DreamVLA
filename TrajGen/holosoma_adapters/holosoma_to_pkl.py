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


# ---- Arm-only table-collision refinement (port of refine_motions.py's hand<->table cost to 29-DOF) ----
# The legacy PyRoki refine kept the RIGHT HAND out of the table half-space {x > x_edge, z < OFFSET_Z}
# during the pre-grab approach and floored the hand height post-grab, optimizing ONLY the right-arm
# joints (legs/left-arm frozen). This is that fix, adapted to the 29-DOF holosoma output. It does NOT
# touch legs -> locomotion/leg-table collisions are out of scope (same as the legacy pipeline).
REFINE_ARM = os.environ.get("HS_REFINE_ARM", "1") == "1"
OFFSET_Z = 0.86              # table-top z (legacy refine_motions.py OFFSET_Z)
OFFSET_X = -0.35             # table near-edge x relative to grab (legacy)
WRIST_TO_COLLISION = 0.35    # legacy grab_pos[0] += this before edge = grab_x + WRIST_TO_COLLISION + OFFSET_X
HAND_TIP_OFFSET = 0.3        # fingertip local-x offset from the hand link (legacy)
REFINE_STEPS = int(os.environ.get("HS_REFINE_STEPS", "1500"))
REFINE_W_REG = 3.0           # keep the arm near its original trajectory (preserve the reach)
REFINE_W_TABLE = 10.0        # table-penetration weight (legacy used 10x on the segment cost)
REFINE_W_FLOOR = 2.0         # post-grab hand-height floor weight
_RARM = list(range(22, 29)) if DOF == 29 else list(range(20, 27))   # right-arm joint indices


def refine_right_arm_table(joints, base_pos, base_quat, grab_idx, chain):
    """Optimize the right-arm joints so the hand tip + hand-origin stay OUT of the table
    half-space during the pre-grab approach (penetration-depth penalty into {x>x_edge, z<OFFSET_Z}),
    with a post-grab height floor, regularized to the original arm so the reach is preserved.
    Returns a new (F,DOF) joints array with only the right-arm columns modified."""
    import torch as _t
    F = joints.shape[0]
    jt = _t.tensor(joints, dtype=_t.float32)
    bp = _t.tensor(base_pos, dtype=_t.float32)
    R = _t.tensor(quat_wxyz_to_R(base_quat), dtype=_t.float32)          # (F,3,3) root->world
    arm0 = jt[:, _RARM].clone()
    # x_edge from the wrist world-x at grab (legacy: grab_x + WRIST_TO_COLLISION + OFFSET_X)
    with _t.no_grad():
        fk0 = chain.forward_kinematics(jt)
        wl = fk0["right_wrist_pitch_link"].get_matrix()[:, :3, 3]        # (F,3) root frame
        wrist_w = _t.einsum("fij,fj->fi", R, wl) + bp
        x_edge = float(wrist_w[grab_idx, 0]) + WRIST_TO_COLLISION + OFFSET_X
    z_table = OFFSET_Z
    tip_local = _t.tensor([HAND_TIP_OFFSET, 0.0, 0.0])
    arm = _t.nn.Parameter(arm0.clone())
    opt = _t.optim.Adam([arm], lr=1e-3)

    def _pen(pts, gi):   # penetration depth into the table corner {x>x_edge, z<z_table}, pre-grab
        p = pts[:gi]
        d = _t.minimum(_t.relu(p[:, 0] - x_edge), _t.relu(z_table - p[:, 2]))
        return (d * d).sum()

    _verbose = os.environ.get("HS_REFINE_VERBOSE", "0") == "1"
    last = 0.0; _tab0 = 0.0; _grad0 = 0.0; _ran = 0
    for _step in range(REFINE_STEPS):
        opt.zero_grad()
        j = jt.clone(); j[:, _RARM] = arm
        fk = chain.forward_kinematics(j)
        hm = fk["right_rubber_hand"].get_matrix()                       # (F,4,4) root frame
        hp = hm[:, :3, 3]; hr = hm[:, :3, :3]
        tip_root = hp + _t.einsum("fij,j->fi", hr, tip_local)
        tip_w = _t.einsum("fij,fj->fi", R, tip_root) + bp              # fingertip world
        orig_w = _t.einsum("fij,fj->fi", R, hp) + bp                   # hand-origin world
        _tab = _pen(tip_w, grab_idx) + _pen(orig_w, grab_idx)
        cost = REFINE_W_TABLE * _tab
        cost = cost + REFINE_W_REG * ((arm - arm0) ** 2).sum(dim=1).mean()
        if grab_idx < F:                                               # post-grab hand-height floor
            cost = cost + REFINE_W_FLOOR * (_t.relu(z_table - tip_w[grab_idx:, 2]) ** 2).mean()
        cost.backward(); _gn = float(arm.grad.norm()) if arm.grad is not None else -1.0
        opt.step(); last = float(cost.item()); _ran = _step + 1
        if _step == 0:
            _tab0 = float(_tab.item()); _grad0 = _gn
        if _verbose and (_step in (0, 1, 10, 50, 200, 500, 1000) or _step == REFINE_STEPS - 1):
            print(f"[refine-arm]   step {_step:4d}  cost={last:.5f}  table_pen={float(_tab.item()):.5f}  |grad|={_gn:.5f}")
        if last < 1e-4:
            break
    out = joints.copy()
    out[:, _RARM] = arm.detach().numpy()
    max_move = float(np.abs(out[:, _RARM] - joints[:, _RARM]).max())
    print(f"[refine-arm] x_edge={x_edge:.3f} z_table={z_table} ran {_ran}/{REFINE_STEPS} steps "
          f"init_table_pen={_tab0:.5f} init|grad|={_grad0:.5f} final_cost={last:.4f} max_arm_move={max_move:.3f} rad")
    return out


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

# --- arm table-collision refine (right arm out of the table during the pre-grab reach) ---
# Runs on the grounded core motion, before the grab-hold/lead-in so grab_idx is the core index.
# HS_REFINE_MODE=al (default): full augmented-Lagrangian refine (refine_al_29 = ported
# refine_motions_al.py: swept-quad table, tip/wrist speed-accel-jerk, hard DOF speed limits,
# laziness, outward-swing approach shaping) -> smooth reaches, no fast punch-through.
# HS_REFINE_MODE=simple: the lightweight per-frame penetration refine (refine_right_arm_table).
if REFINE_ARM and 0 < grab_idx < F:
    _mode = os.environ.get("HS_REFINE_MODE", "al")
    if _mode == "al":
        import refine_al_29
        joints = refine_al_29.refine_arm(joints, base_pos, base_quat, obj_pos[grab_idx], grab_idx, fps=20.0)
    else:
        joints = refine_right_arm_table(joints, base_pos, base_quat, grab_idx, chain)

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
# grab_pos IS the true holosoma object position (the bottle the hand was retargeted to grasp),
# so motion_lib can use its xy directly instead of the wrist_yaw+fixed-offset heuristic (which
# mis-places the object ~10-16 cm laterally for holosoma wrist poses). Flagged for motion_lib.
pkl = {"global_pose": global_pose, "joints": torch.tensor(joints, dtype=torch.float32),
       "global_position": torch.tensor(base_pos, dtype=torch.float32),
       "grab_pos": grab_pos, "grab_idx": grab_idx, "grab_pos_is_object": True}
with open(out_base + ".pkl", "wb") as f:
    pickle.dump(pkl, f)
_wr = np.linalg.norm(joints[:, 13]) if DOF == 29 else 0.0
_wp = np.linalg.norm(joints[:, 14]) if DOF == 29 else 0.0
print(f"wrote {out_base}.pkl  {DOF}-DOF  frames {F}->{joints.shape[0]} (lead-in {PAUSE}+{INTERP}) "
      f"grab_idx={grab_idx}  ground_shift[{mz.min():.3f},{mz.max():.3f}]  "
      f"waist_roll|pitch L2={_wr:.3f}|{_wp:.3f}  frame0 |dq/dt|@20fps="
      f"{np.linalg.norm((joints[1]-joints[0])/0.05):.3f} (want ~0)")
