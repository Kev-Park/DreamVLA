"""Stitch a static-spawn pick (e.g. Holosoma_Pick_29_latband60) hand trajectory into a walking pick.

For each clip id present in both pools:
  1. keep the walking clip up to its FINAL FOOTFALL (first frame from which both ankles stay
     planted through the grab: within 2 cm of their lowest height and moving < 5 mm/frame);
  2. from there on the lower body and root are HELD at the footfall pose (legs + root frozen), and
     the clip is EXTENDED: TRANSITION frames, then the static clip from its hand-motion onset to
     its end (approach, grab hold, lift) -- so the static approach plays at its own timing;
  3. waist + right arm are re-solved by IK (blended in over TRANSITION frames from the walking
     pose) so the right_rubber_hand follows the static clip's hand pose (position + orientation)
     and the torso its torso orientation, both expressed relative to the bottle and robot heading,
     then re-anchored onto the walking clip's bottle and footfall heading;
  4. object poses: walking object until the stitch, then the static clip's object trajectory
     under the same re-anchoring; grab_idx moves to the stitched grab (its FREEZE_FOR hold is
     carried over from the static clip).
The left arm is frozen in both pools (holosoma_to_pkl freeze_left_arm) and is held unchanged.

Clips whose footfall body pose is further than MAX_POS_ERR / MAX_YAW_ERR from the static spawn
(bottle-relative) are skipped: the static hand path would then be a different reach.

usage: python stitch_static_approach.py <walk_pool> <static_pool> <out_pool> [pick_id ...]
pools are directory names under ~/kevin/ref_motions (REF_MOTIONS_DIR overrides) or paths.
"""
import os, sys, glob, pickle
import numpy as np, torch
import jaxlie, jax.numpy as jnp
import pytorch_kinematics as pk

TRANSITION = 15            # frames (20 fps) to blend waist/right arm from the walking pose into the static path
ONSET_MOVE = 0.03          # m, static hand displacement from its rest pose that marks the approach onset
PLANT_BAND, PLANT_SPEED = 0.02, 0.005
MAX_POS_ERR, MAX_YAW_ERR = 0.10, np.radians(15.0)
IK_ITERS = 800

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "..", "..", "Training", "HumanoidVerse", "humanoidverse", "data", "robots", "g1", "g1_29dof.urdf")
LEGS, WAIST, RARM = slice(0, 12), slice(12, 15), slice(22, 29)
HAND = "right_rubber_hand"


def pool_dir(p):
    if os.path.isdir(p):
        return p
    root = os.environ.get("REF_MOTIONS_DIR", os.path.expanduser("~/kevin/ref_motions"))
    return os.path.join(root, os.path.basename(p.rstrip("/")))


def quat_to_R(q):    # (N,4) wxyz -> (N,3,3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def yaw_of(q):
    return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def quat_mul(a, b):  # wxyz, broadcast
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def load(f, chain):
    d = pickle.load(open(f, "rb"))
    rot = d["global_pose"].rotation()
    Q = np.asarray(rot.wxyz if not callable(getattr(rot, "wxyz", None)) else rot.wxyz(), dtype=np.float64)
    c = dict(J=np.asarray(d["joints"], dtype=np.float64), P=np.asarray(d["global_pose"].translation(), dtype=np.float64),
             Q=Q, O=np.asarray(d["object_poses"], dtype=np.float64), g=int(d["grab_idx"]))
    fk = chain.forward_kinematics(torch.tensor(c["J"], dtype=torch.float32))
    Rr = quat_to_R(Q)
    for name, key in (("left_ankle_roll_link", "la"), ("right_ankle_roll_link", "ra"), (HAND, "hand"),
                      ("torso_link", "torso")):
        M = fk[name].get_matrix().double().numpy()
        c[key] = np.einsum("nij,nj->ni", Rr, M[:, :3, 3]) + c["P"]
        c[key + "R"] = Rr @ M[:, :3, :3]
    return c


def final_footfall(c):
    def planted(a):
        sp = np.r_[0.0, np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1)]
        return (a[:, 2] < a[:, 2].min() + PLANT_BAND) & (sp < PLANT_SPEED)
    both = planted(c["la"]) & planted(c["ra"])
    for k in range(c["g"], -1, -1):
        if not both[k]:
            return min(k + 1, len(c["J"]) - 1)
    return 0


def body_offset(c, k, w):
    """root position / bearing-to-object relative to the object at frame w, in the root heading frame at k."""
    yw = yaw_of(c["Q"][k]); F = np.array([np.cos(yw), np.sin(yw)]); L = np.array([-np.sin(yw), np.cos(yw)])
    r = c["P"][k, :2] - c["O"][w, :2]
    b = -r
    return np.array([r @ F, r @ L]), np.arctan2(F[0] * b[1] - F[1] * b[0], F @ b)


def solve_upper(chain, lim, Jfix, q0, root_P, root_R, tgt_p, tgt_R, torso_R):
    """Waist + right-arm IK for a batch of frames (q = [waist 3, right arm 7]); Jfix supplies legs and
    left arm (N,29). The walking footfall pelvis is not level (roll/pitch up to ~9 deg) while the
    static clip's is, so the waist is solved too, toward the static clip's TORSO orientation: the
    torso then stands as it did in the static clip and the arm reproduces its reach."""
    J = torch.tensor(Jfix, dtype=torch.float32)
    q = torch.tensor(q0, dtype=torch.float32).clone().requires_grad_(True)
    q_init = q.detach().clone()
    Rr = torch.tensor(root_R, dtype=torch.float32); Pr = torch.tensor(root_P, dtype=torch.float32)
    tp = torch.tensor(tgt_p, dtype=torch.float32); tR = torch.tensor(tgt_R, dtype=torch.float32)
    tT = torch.tensor(torso_R, dtype=torch.float32)
    idx = list(range(12, 15)) + list(range(22, 29))
    lo, hi = (torch.tensor(x[idx], dtype=torch.float32) for x in lim)
    opt = torch.optim.Adam([q], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, IK_ITERS, eta_min=1e-3)
    for _ in range(IK_ITERS):
        Jf = torch.cat([J[:, :12], q[:, :3], J[:, 15:22], q[:, 3:]], dim=1)
        fk = chain.forward_kinematics(Jf)
        M = fk[HAND].get_matrix()
        p = torch.einsum("nij,nj->ni", Rr, M[:, :3, 3]) + Pr
        R = Rr @ M[:, :3, :3]
        RT = Rr @ fk["torso_link"].get_matrix()[:, :3, :3]
        e_p = ((p - tp) ** 2).sum(-1)
        e_r = 3.0 - (R * tR).sum((-1, -2))                      # = 2(1 - cos angle)
        e_t = 3.0 - (RT * tT).sum((-1, -2))
        smooth = ((q[2:] - 2 * q[1:-1] + q[:-2]) ** 2).sum(-1).mean() if len(q) > 2 else q.sum() * 0
        loss = (1000.0 * e_p + 10.0 * e_r + 1.0 * e_t).mean() + 1e-3 * ((q - q_init) ** 2).sum(-1).mean() + 10.0 * smooth
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            q.copy_(torch.max(torch.min(q, hi), lo))
    ang = lambda e: torch.arccos(torch.clamp((2.0 - e.detach()) / 2.0, -1, 1)).numpy()
    return q.detach().double().numpy(), e_p.detach().sqrt().numpy(), ang(e_r), ang(e_t)


def smoothstep(n):
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return t * t * (3 - 2 * t)


def stitch(w, s, chain, lim):
    fw = final_footfall(w); gs = s["g"]; gw = w["g"]
    h0 = s["hand"][min(20, gs)]
    on = next((k for k in range(min(20, gs), gs) if np.linalg.norm(s["hand"][k] - h0) > ONSET_MOVE), gs)
    dw, yw_w = body_offset(w, fw, gw); ds, yw_s = body_offset(s, gs, gs)
    dpos = np.linalg.norm(dw - ds); dyaw_err = abs(np.angle(np.exp(1j * (yw_w - yw_s))))
    info = dict(footfall=fw, onset=on, dpos=dpos, dyaw=np.degrees(dyaw_err))
    if dpos > MAX_POS_ERR or dyaw_err > MAX_YAW_ERR:
        return None, info

    # rigid re-anchoring of the static clip: its grab-frame heading + bottle -> walking footfall heading + bottle
    dyaw = yaw_of(w["Q"][fw]) - yaw_of(s["Q"][gs])
    c_, s_ = np.cos(dyaw), np.sin(dyaw)
    Rz = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
    qz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
    a_s, a_w = s["O"][gs, :3], w["O"][gw, :3]
    mp = lambda x: (x - a_s) @ Rz.T + a_w

    seg = np.arange(on, len(s["J"]))                    # static frames played after the transition
    n_new = TRANSITION + len(seg)
    tgt_p = np.concatenate([np.repeat(mp(s["hand"][on:on + 1]), TRANSITION, 0), mp(s["hand"][seg])])
    tgt_R = np.concatenate([np.repeat(Rz @ s["handR"][on:on + 1], TRANSITION, 0), Rz @ s["handR"][seg]])
    tor_R = np.concatenate([np.repeat(Rz @ s["torsoR"][on:on + 1], TRANSITION, 0), Rz @ s["torsoR"][seg]])

    # body over the new section: legs/root/left arm held at the footfall; waist + right arm by IK
    Jn = np.repeat(w["J"][fw:fw + 1], n_new, 0)
    up = np.concatenate([s["J"][:, WAIST], s["J"][:, RARM]], 1)
    q0 = np.concatenate([np.repeat(up[on:on + 1], TRANSITION, 0), up[seg]])
    Pn = np.repeat(w["P"][fw:fw + 1], n_new, 0); Qn = np.repeat(w["Q"][fw:fw + 1], n_new, 0)
    # IK the static path (incl. the held start pose); the transition is then blended in joint space
    qa, ep, er, et = solve_upper(chain, lim, Jn, q0, Pn, quat_to_R(Qn), tgt_p, tgt_R, tor_R)
    b = smoothstep(TRANSITION)[:, None]
    q_w = np.concatenate([w["J"][fw, WAIST], w["J"][fw, RARM]])
    qa[:TRANSITION] = (1 - b) * q_w + b * qa[TRANSITION]
    Jn[:, WAIST] = qa[:, :3]; Jn[:, RARM] = qa[:, 3:]

    On = np.concatenate([np.repeat(w["O"][fw:fw + 1], TRANSITION, 0),
                         np.concatenate([mp(s["O"][seg, :3]), quat_mul(qz, s["O"][seg, 3:])], 1)])
    J = np.concatenate([w["J"][:fw], Jn]); P = np.concatenate([w["P"][:fw], Pn])
    Q = np.concatenate([w["Q"][:fw], Qn]); O = np.concatenate([w["O"][:fw], On])
    g_new = fw + TRANSITION + (gs - on)
    k_g = g_new - fw
    info.update(n_old=len(w["J"]), n_new=len(J), g_old=gw, g_new=g_new,
                ik_pos_cm=100 * float(ep[TRANSITION:k_g + 1].max()),
                ik_pos_cm_post=100 * float(ep[k_g:].max()),
                ik_rot_deg=float(np.degrees(er[TRANSITION:k_g + 1].max())),
                torso_deg=float(np.degrees(et[TRANSITION:].max())),
                obj_jump_cm=100 * float(np.linalg.norm(On[TRANSITION, :3] - w["O"][fw, :3])))
    pkl = {"global_pose": jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(Q)), jnp.array(P)),
           "joints": torch.tensor(J, dtype=torch.float32),
           "global_position": torch.tensor(P, dtype=torch.float32),
           "grab_pos": torch.tensor(O[g_new, :3], dtype=torch.float32), "grab_idx": int(g_new),
           "grab_pos_is_object": True, "object_poses": torch.tensor(O, dtype=torch.float32)}
    return pkl, info


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    wdir, sdir, odir = pool_dir(sys.argv[1]), pool_dir(sys.argv[2]), pool_dir(sys.argv[3])
    ids = sys.argv[4:]
    chain = pk.build_chain_from_urdf(open(URDF, "rb").read()).to(dtype=torch.float32)
    lim = [np.asarray(x, dtype=np.float64) for x in chain.get_joint_limits()]
    W = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(wdir, "pick_*.pkl"))}
    S = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(sdir, "pick_*.pkl"))}
    names = sorted(set(W) & set(S), key=lambda x: int(x.split("_")[1]))
    if ids:
        names = [n for n in names if n.split("_")[1] in ids or n in ids]
    os.makedirs(odir, exist_ok=True)
    print("%-9s %4s %4s %5s %6s %9s %9s %9s %11s %10s %9s %8s" % ("clip", "ff", "on", "dpos", "dyaw", "len",
          "grab", "ik_pos_cm", "ik_post_cm", "ik_rot_deg", "torso_deg", "objjump"))
    nw = 0
    for n in names:
        pkl, info = stitch(load(W[n], chain), load(S[n], chain), chain, lim)
        if pkl is None:
            print("%-9s %4d %4d %5.1f %6.1f  SKIPPED (footfall body pose off the static spawn)"
                  % (n, info["footfall"], info["onset"], 100 * info["dpos"], info["dyaw"]))
            continue
        with open(os.path.join(odir, n + ".pkl"), "wb") as f:
            pickle.dump(pkl, f)
        nw += 1
        print("%-9s %4d %4d %5.1f %6.1f %3d->%3d %3d->%3d %9.2f %11.2f %10.1f %9.1f %8.2f" % (
            n, info["footfall"], info["onset"], 100 * info["dpos"], info["dyaw"], info["n_old"], info["n_new"],
            info["g_old"], info["g_new"], info["ik_pos_cm"], info["ik_pos_cm_post"], info["ik_rot_deg"],
            info["torso_deg"], info["obj_jump_cm"]))
    print("wrote %d/%d clips -> %s" % (nw, len(names), odir))


if __name__ == "__main__":
    main()
