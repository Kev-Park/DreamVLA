"""Diagnostic for the Adapter B arm-refine loop (29-DOF validation + tunneling check).

Reproduces refine_right_arm_table on one holosoma .npz with full instrumentation:
  - verifies the 29-DOF right-arm joint indices (_RARM) map to the right-arm joint NAMES,
  - verifies the FK hand/wrist links resolve and sit where expected,
  - logs the optimization (table/reg/floor cost components + gradient norm per step),
  - dumps the fingertip world (x,z) trajectory in the pre-grab window BEFORE vs AFTER,
  - flags per-frame penetration AND between-frame tunneling (segment crossing the table box)
    so we can see whether the (per-frame) cost leaves swept-path tunneling behind.

Usage: python refine_diag.py <holosoma_out.npz>
"""
import os, sys, numpy as np, torch

DOF = 29
JN = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
_RARM = list(range(22, 29))
LARM0 = 15
OFFSET_Z = 0.86; OFFSET_X = -0.35; WRIST_TO_COLLISION = 0.35; HAND_TIP_OFFSET = 0.3
REFINE_STEPS = 1500; W_REG = 3.0; W_TABLE = 10.0; W_FLOOR = 2.0
import pytorch_kinematics as pk
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Training", "HumanoidVerse", "humanoidverse", "data", "robots", "g1", "g1_29dof.urdf")


def quat_wxyz_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-w*z); R[:, 0, 2] = 2*(x*z+w*y)
    R[:, 1, 0] = 2*(x*y+w*z); R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-w*x)
    R[:, 2, 0] = 2*(x*z-w*y); R[:, 2, 1] = 2*(y*z+w*x); R[:, 2, 2] = 1-2*(x*x+y*y)
    return R


npz = sys.argv[1]
q = np.array(np.load(npz)["qpos"]).astype(np.float64)
F = q.shape[0]
base_pos = q[:, 0:3].copy(); base_quat = q[:, 3:7].copy(); joints = q[:, 7:36].copy(); obj_pos = q[:, 36:39].copy()
grab_idx = int(max(0, np.argmax(np.linalg.norm(obj_pos - obj_pos[0][None], axis=1) > 1e-4) - 1))

chain = pk.build_chain_from_urdf(open(URDF, "rb").read())
# ground + freeze left arm (Adapter B parity)
fk = chain.forward_kinematics(torch.tensor(joints, dtype=torch.float32))
local = torch.stack([fk[l].get_matrix()[:, :3, 3] for l in fk], dim=1).numpy()
mz = (np.einsum("fij,flj->fli", quat_wxyz_to_R(base_quat), local) + base_pos[:, None, :])[:, :, 2].min(axis=1)
base_pos[:, 2] -= mz
joints[:, LARM0:LARM0+7] = 0.0; joints[:, LARM0] = 0.35; joints[:, LARM0+1] = 0.16; joints[:, LARM0+3] = 0.87

print("=" * 70)
print(f"npz={os.path.basename(npz)}  F={F}  grab_idx={grab_idx}")
print("--- 29-DOF joint mapping ---")
print(f"_RARM idx {_RARM} -> {[JN[i] for i in _RARM]}")
assert all('right' in JN[i] for i in _RARM), "RARM indices are NOT all right-arm!"
print(f"left-arm frozen block LARM0={LARM0} -> {[JN[i] for i in range(LARM0, LARM0+7)]}")
fk_names = list(chain.forward_kinematics(torch.tensor(joints[:1], dtype=torch.float32)).keys())
print(f"FK links: {len(fk_names)}  has right_rubber_hand={'right_rubber_hand' in fk_names}  has right_wrist_pitch_link={'right_wrist_pitch_link' in fk_names}")

# ---- instrumented refine ----
jt = torch.tensor(joints, dtype=torch.float32)
bp = torch.tensor(base_pos, dtype=torch.float32)
R = torch.tensor(quat_wxyz_to_R(base_quat), dtype=torch.float32)
arm0 = jt[:, _RARM].clone()
with torch.no_grad():
    wl = chain.forward_kinematics(jt)["right_wrist_pitch_link"].get_matrix()[:, :3, 3]
    wrist_w = torch.einsum("fij,fj->fi", R, wl) + bp
    x_edge = float(wrist_w[grab_idx, 0]) + WRIST_TO_COLLISION + OFFSET_X
z_table = OFFSET_Z
tip_local = torch.tensor([HAND_TIP_OFFSET, 0.0, 0.0])
print(f"--- table half-space: x > {x_edge:.3f} AND z < {z_table} ---")


def tips_world(arm_vals):
    j = jt.clone(); j[:, _RARM] = arm_vals
    hm = chain.forward_kinematics(j)["right_rubber_hand"].get_matrix()
    hp = hm[:, :3, 3]; hr = hm[:, :3, :3]
    tip_root = hp + torch.einsum("fij,j->fi", hr, tip_local)
    return (torch.einsum("fij,fj->fi", R, tip_root) + bp)


def pen(pts, gi):
    p = pts[:gi]
    d = torch.minimum(torch.relu(p[:, 0] - x_edge), torch.relu(z_table - p[:, 2]))
    return (d * d).sum()


def tunnel_count(tips_np, gi):
    """between-frame tunneling: segment [t,t+1] enters the table box {x>x_edge, z<z_table}
    even if both endpoints are outside it (per-frame check would miss)."""
    cnt = 0
    for t in range(min(gi, len(tips_np) - 1)):
        a, b = tips_np[t], tips_np[t + 1]
        seg_inside = False
        for s in np.linspace(0, 1, 11):
            p = a + s * (b - a)
            if p[0] > x_edge and p[2] < z_table:
                seg_inside = True; break
        ep_inside = (a[0] > x_edge and a[2] < z_table) or (b[0] > x_edge and b[2] < z_table)
        if seg_inside and not ep_inside:
            cnt += 1
    return cnt


tips0 = tips_world(arm0).detach().numpy()
arm = torch.nn.Parameter(arm0.clone())
opt = torch.optim.Adam([arm], lr=1e-3)
print("--- optimization trace ---")
for step in range(REFINE_STEPS):
    opt.zero_grad()
    tip_w = tips_world(arm)
    j = jt.clone(); j[:, _RARM] = arm
    hm = chain.forward_kinematics(j)["right_rubber_hand"].get_matrix()
    orig_w = torch.einsum("fij,fj->fi", R, hm[:, :3, 3]) + bp
    c_table = pen(tip_w, grab_idx) + pen(orig_w, grab_idx)
    c_reg = ((arm - arm0) ** 2).sum(dim=1).mean()
    c_floor = (torch.relu(z_table - tip_w[grab_idx:, 2]) ** 2).mean() if grab_idx < F else torch.tensor(0.0)
    cost = W_TABLE * c_table + W_REG * c_reg + W_FLOOR * c_floor
    cost.backward(); gn = float(arm.grad.norm()); opt.step()
    if step in (0, 1, 25, 100, 300, 700, REFINE_STEPS - 1):
        print(f"  step {step:4d}  cost={float(cost):.5f}  table={float(c_table):.5f}  reg={float(c_reg):.5f}  floor={float(c_floor):.5f}  |grad|={gn:.4f}")
    if float(cost) < 1e-4: break

tips1 = tips_world(arm.detach()).detach().numpy()
gi = grab_idx
# per-frame penetration frames (pre-grab)
pf0 = int(((tips0[:gi, 0] > x_edge) & (tips0[:gi, 2] < z_table)).sum())
pf1 = int(((tips1[:gi, 0] > x_edge) & (tips1[:gi, 2] < z_table)).sum())
print("--- fingertip pre-grab, table-clearance ---")
print(f"per-frame frames INSIDE table (x>edge & z<{z_table}):  before={pf0}  after={pf1}")
print(f"between-frame TUNNELING segments (miss per-frame):     before={tunnel_count(tips0, gi)}  after={tunnel_count(tips1, gi)}")
print(f"max arm move: {float(np.abs(arm.detach().numpy() - arm0.numpy()).max()):.3f} rad")
# dump fingertip z around grab
w = range(max(0, gi - 12), min(F, gi + 2))
print("frame:  " + " ".join(f"{t:5d}" for t in w))
print("tip_x0: " + " ".join(f"{tips0[t,0]:5.2f}" for t in w))
print("tip_z0: " + " ".join(f"{tips0[t,2]:5.2f}" for t in w) + f"   (edge_x={x_edge:.2f}, top_z={z_table})")
print("tip_z1: " + " ".join(f"{tips1[t,2]:5.2f}" for t in w) + "   (after refine)")
