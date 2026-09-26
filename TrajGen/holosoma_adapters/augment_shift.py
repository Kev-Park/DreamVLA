"""IK augmentation: re-solve a FINISHED reference clip's right arm for a shifted object.

Takes the product .pkl of a reference dataset (e.g. the frozen Holosoma_Pick_29_latband60),
translates the whole object trajectory in the table plane, and re-runs the SAME augmented-
Lagrangian arm refine that produced the clip -- so one clip becomes N variants that grasp a
different point on the table with the lower body, heading and timing untouched.

This is the DreamControl Pick_real1/refine_motions.py augmentation (shift the reach target on a
grid, re-refine, warm-started from the neighbouring solution) applied to the object rather than
to the reference wrist. Warm-starting is free here: refine_al_29 initialises the optimiser from
the joints it is given, and those are already the converged solution for the unshifted object.

Everything the refine and the pkl derive from the object follows the shift automatically:
palm_target and the post-grab palm trajectory, the object-anchored table edge, grab_pos and
object_poses. The lead-in and the FREEZE_FOR grab hold are already baked into the input clip and
are NOT re-applied.

Run it with the SAME HS_* refine stack that produced the input dataset (the refine's windows,
walls, grasp offset and limits are read from the environment by refine_al_29 at import).

  python augment_shift.py --refs Holosoma_Pick_29_latband60 --out Holosoma_Pick_29_latband60aug \
      --shifts "0,0.06; 0,-0.06; 0.05,0; -0.05,0"

  --grid FWD_MIN,FWD_MAX,N_FWD,LEFT_MIN,LEFT_MAX,N_LEFT   generates the shift list instead
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_al_29                                                   # noqa: E402  (reads HS_* at import)

def _pooled(name: str) -> str:
    """Mirror of repo_paths.pooled(): a bare NAME lives in the shared ~/kevin/ref_motions pool."""
    if os.sep in name or "/" in name:
        return name
    root = os.environ.get("REF_MOTIONS_DIR")
    if root:
        return os.path.join(root, name)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (os.path.expanduser("~/kevin/ref_motions"), os.path.join(os.path.dirname(repo), "ref_motions"),
                 os.path.expanduser("~/ref_motions")):
        if os.path.isdir(os.path.join(cand, name)):
            return os.path.join(cand, name)
    return os.path.join(os.path.expanduser("~/kevin/ref_motions"), name)


def _yaw_of(quat_wxyz) -> float:
    w, x, y, z = (float(quat_wxyz[k]) for k in range(4))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _heading_delta(yaw: float, fwd: float, left: float) -> np.ndarray:
    return np.array([np.cos(yaw) * fwd - np.sin(yaw) * left,
                     np.sin(yaw) * fwd + np.cos(yaw) * left])


def _unfreeze(a: np.ndarray, g: int, F: int) -> np.ndarray:
    """Inverse of holosoma_to_pkl.freeze_hold: drop the F duplicated grab-hold frames.

    freeze_hold is length-preserving -- it holds frame g for F extra frames and shifts the tail
    back, dropping the last F frames. The refine must run on the UN-held motion (as it did in the
    original pipeline, where the hold is applied after the refine): a displaced target demands
    motion exactly where the source has a zero-velocity plateau, and the 6 rad/s cap then forces
    the correction into the frames adjacent to the hold. Returns length len(a) - F.
    """
    return np.concatenate([a[:g], a[g + F:]], axis=0)


def _refreeze(a: np.ndarray, g: int, F: int) -> np.ndarray:
    """Re-apply the grab hold to an un-frozen array (inverse of _unfreeze). Returns len(a) + F."""
    return np.concatenate([a[:g], np.repeat(a[g:g + 1], F, axis=0), a[g:]], axis=0)


def parse_shifts(args) -> list[tuple[float, float]]:
    if args.grid:
        f0, f1, nf, l0, l1, nl = (float(v) for v in args.grid.split(","))
        fs = np.linspace(f0, f1, int(nf))
        ls = np.linspace(l0, l1, int(nl))
        return [(float(f), float(l)) for f in fs for l in ls]
    out = []
    for part in args.shifts.split(";"):
        part = part.strip()
        if not part:
            continue
        f, l = (float(v) for v in part.split(","))
        out.append((f, l))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="input dataset (bare pooled name or path)")
    ap.add_argument("--out", required=True, help="output dataset (bare pooled name or path)")
    ap.add_argument("--shifts", default="", help='"fwd,left; fwd,left; ..." in metres, heading frame')
    ap.add_argument("--grid", default="", help="FWD_MIN,FWD_MAX,N_FWD,LEFT_MIN,LEFT_MAX,N_LEFT")
    ap.add_argument("--ids", default="", help="comma-separated clip ids (default: every pick_*.pkl)")
    ap.add_argument("--lat-band", default="", help='"lo,hi" reject a variant whose root->object lateral '
                                                   "offset at grab falls outside this band (metres, + = left)")
    args = ap.parse_args()

    in_dir, out_dir = _pooled(args.refs), _pooled(args.out)
    os.makedirs(out_dir, exist_ok=True)
    if args.ids:
        names = [f"pick_{i.strip()}.pkl" for i in args.ids.split(",")]
    else:
        names = sorted(n for n in os.listdir(in_dir) if n.startswith("pick_") and n.endswith(".pkl"))
    shifts = parse_shifts(args)
    band = tuple(float(v) for v in args.lat_band.split(",")) if args.lat_band else None
    print(f"[augment] {len(names)} clips x {len(shifts)} shifts -> {out_dir}")

    n_ok = n_skip = 0
    for name in names:
        with open(os.path.join(in_dir, name), "rb") as f:
            d = pickle.load(f)
        joints0 = np.asarray(d["joints"], dtype=np.float32)
        base_pos = np.asarray(d["global_position"], dtype=np.float32)
        base_quat = np.asarray(d["global_pose"].rotation().wxyz, dtype=np.float32)
        obj0 = np.asarray(d["object_poses"], dtype=np.float32)          # (F,7) pos+quat
        grab_idx = int(d["grab_idx"])
        yaw = _yaw_of(base_quat[grab_idx])
        hdg = np.array([np.cos(yaw), np.sin(yaw)])
        lat_ax = np.array([-np.sin(yaw), np.cos(yaw)])
        # grab_pos is sampled _lead = PAUSE+INTERP frames BEFORE the grab (holosoma_to_pkl:365);
        # the object is static there, so this is just its rest position.
        lead = 0 if os.environ.get("HS_NO_LEADIN", "0") == "1" else 20
        # The product clip carries the FREEZE_FOR grab hold; strip it for the refine and restore it
        # afterwards, so the construction order matches the base pipeline (refine, then hold).
        F = int(os.environ.get("HS_FREEZE_FOR", "10"))
        if F > 0 and not np.allclose(joints0[grab_idx], joints0[grab_idx + F], atol=1e-6):
            print(f"  {name}: no {F}-frame grab hold detected at {grab_idx}; refining as-is")
            F = 0
        j_src = _unfreeze(joints0, grab_idx, F) if F else joints0
        bp = _unfreeze(base_pos, grab_idx, F) if F else base_pos
        bq = _unfreeze(base_quat, grab_idx, F) if F else base_quat

        for (fwd, left) in shifts:
            tag = f"f{int(round(fwd * 1000)):+d}l{int(round(left * 1000)):+d}".replace("+", "p").replace("-", "m")
            out_name = f"{name[:-4]}_{tag}.pkl"
            obj = obj0.copy()
            delta2 = _heading_delta(yaw, fwd, left)
            obj[:, :2] += delta2[None, :]
            d_lat = float(np.dot(obj[grab_idx, :2] - base_pos[grab_idx, :2], lat_ax))
            d_along = float(np.dot(obj[grab_idx, :2] - base_pos[grab_idx, :2], hdg))
            if band is not None and not (band[0] <= d_lat <= band[1]):
                print(f"  {out_name}: SKIP lateral {d_lat:+.3f} outside [{band[0]:+.2f},{band[1]:+.2f}]")
                n_skip += 1
                continue
            print(f"  {out_name}: root->object along {d_along:+.3f} lateral {d_lat:+.3f}")
            obj_ref = _unfreeze(obj, grab_idx, F) if F else obj
            joints = refine_al_29.refine_arm(
                j_src.copy(), bp, bq, obj_ref[grab_idx, :3], grab_idx, fps=20.0,
                obj_traj=obj_ref[:, :3],
                src_joints=j_src, palm_shift=np.array([delta2[0], delta2[1], 0.0]))
            if not bool(np.isfinite(joints).all()):
                print(f"  {out_name}: SKIP non-finite refine output")
                n_skip += 1
                continue
            if F:
                joints = _refreeze(joints, grab_idx, F)
            _dq = float(np.abs(joints[:, 22:29] - joints0[:, 22:29]).max())
            print(f"  {out_name}: max |dq vs source| {_dq:.3f} rad")
            pkl = {"global_pose": d["global_pose"],
                   "joints": torch.tensor(joints, dtype=torch.float32),
                   "global_position": torch.tensor(base_pos, dtype=torch.float32),
                   "grab_pos": torch.tensor(obj[max(0, grab_idx - lead), :3], dtype=torch.float32),
                   "grab_idx": grab_idx, "grab_pos_is_object": True,
                   "object_poses": torch.tensor(obj, dtype=torch.float32)}
            with open(os.path.join(out_dir, out_name), "wb") as f:
                pickle.dump(pkl, f)
            n_ok += 1
    print(f"[augment] wrote {n_ok} variants, skipped {n_skip} -> {out_dir}")


if __name__ == "__main__":
    main()
