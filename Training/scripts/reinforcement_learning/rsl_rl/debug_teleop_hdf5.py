"""Post-implementation debug checks for /teleop HDF5 schema. DELETE AFTER USE.

Usage:
    python debug_teleop_hdf5.py --hdf5 <path1> [<path2> ...]

Single file  -> runs [DEBUG-1] schema, [DEBUG-2] SE(3) validity,
                [DEBUG-3] frame-0 sanity, [DEBUG-4] locomotion invariance,
                [DEBUG-5] FK round-trip (numpy-only, self-consistent).
Two files    -> additionally runs [DEBUG-6] finger layout stability.
Two files    -> additionally runs [DEBUG-7] determinism hash compare.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------- helpers ----------

def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quaternion -> 3x3 rotation matrix. Matches isaaclab convention."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w,x,y,z) quaternion."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


# ---------- per-file checks ----------

def debug_1_schema(path: Path) -> None:
    print(f"===== [DEBUG-1] SCHEMA ({path.name}) =====")
    with h5py.File(path, "r") as f:
        def _walk(name: str, obj) -> None:
            kind = "G" if isinstance(obj, h5py.Group) else f"D{getattr(obj, 'shape', '')}"
            print(f"  [{kind}] {name}")
        print("FULL_STRUCTURE:")
        f.visititems(_walk)
        print(f"ROOT_ATTRS_KEYS: {list(f.attrs.keys())}")
        print(f"HAS_TELEOP_GROUP:             {'teleop' in f}")
        print(f"HAS_STATE_RAW_TELEOP (want F): {'state/raw/teleop' in f}")
        print(f"HAS_STATE_GROUP:              {'state' in f}")
        if "teleop" not in f:
            print("no /teleop group -> aborting schema check")
            return
        g = f["teleop"]
        print(f"LW_SHAPE:  {g['left_wrist'].shape}  LW_DTYPE:  {g['left_wrist'].dtype}")
        print(f"RW_SHAPE:  {g['right_wrist'].shape}  RW_DTYPE:  {g['right_wrist'].dtype}")
        print(f"TS_SHAPE:  {g['timestamps'].shape}  TS_DTYPE:  {g['timestamps'].dtype}")
        print(f"ATTRS:     {dict(g.attrs)}")
        if "state/raw/robot/joint_pos" in f:
            print(f"N_JOINT_POS (state/raw/robot/joint_pos): {f['state/raw/robot/joint_pos'].shape[0]}")
        else:
            print("state/raw/robot/joint_pos not present")
        if "teleop/calibration" in f:
            print(
                f"CAL LW_SHAPE: {f['teleop/calibration/left_wrist'].shape}  "
                f"CAL RW_SHAPE: {f['teleop/calibration/right_wrist'].shape}"
            )
        if "teleop/finger_joints" in f:
            fj = f["teleop/finger_joints"]
            print(
                f"FJ LEFT_SHAPE:  {fj['left'].shape}  "
                f"FJ RIGHT_SHAPE: {fj['right'].shape}"
            )


def debug_2_se3(path: Path) -> None:
    print(f"===== [DEBUG-2] SE(3) VALIDITY ({path.name}) =====")
    with h5py.File(path, "r") as f:
        if "teleop" not in f:
            print("no /teleop group -> skip")
            return
        for side in ("left_wrist", "right_wrist"):
            T = f[f"teleop/{side}"][...]
            R = T[:, :3, :3]
            det = np.linalg.det(R)
            gram = np.einsum("nij,nkj->nik", R, R)
            ortho_err = float(np.max(np.abs(gram - np.eye(3)[None])))
            bottom_err = float(np.max(np.abs(T[:, 3, :] - np.array([0, 0, 0, 1.0]))))
            print(
                f"{side:12s} det_min={float(det.min()):+.9f} "
                f"det_max={float(det.max()):+.9f} "
                f"ortho_err={ortho_err:.3e} bottom_err={bottom_err:.3e}"
            )


def debug_3_frame0(path: Path) -> None:
    print(f"===== [DEBUG-3] FRAME-0 SANITY ({path.name}) =====")
    with h5py.File(path, "r") as f:
        if "teleop" not in f:
            print("no /teleop group -> skip")
            return
        lw0 = f["teleop/left_wrist"][0]
        rw0 = f["teleop/right_wrist"][0]
        print(f"LW0_TRANS (pelvis-frame, m): {lw0[:3, 3].tolist()}")
        print(f"RW0_TRANS (pelvis-frame, m): {rw0[:3, 3].tolist()}")
        print(f"|LW0_TRANS|: {float(np.linalg.norm(lw0[:3, 3])):.4f}  "
              f"|RW0_TRANS|: {float(np.linalg.norm(rw0[:3, 3])):.4f}")
        if "state/raw/robot/root_pos_w" in f:
            root0 = f["state/raw/robot/root_pos_w"][0]
            print(f"ROOT0_POS_W (world, m):      {root0.tolist()}")
            print(
                "(LW0/RW0 should be robot-scale ~O(0.1-0.5m), NOT close to ROOT0_POS_W"
                " — if they match ROOT0_POS_W you're emitting world-frame.)"
            )


def debug_4_locomotion_invariance(path: Path) -> None:
    print(f"===== [DEBUG-4] LOCOMOTION INVARIANCE ({path.name}) =====")
    with h5py.File(path, "r") as f:
        if "teleop" not in f:
            print("no /teleop group -> skip")
            return
        lw = f["teleop/left_wrist"][...]
        rw = f["teleop/right_wrist"][...]
        lt0, lt_last = lw[0, :3, 3], lw[-1, :3, 3]
        rt0, rt_last = rw[0, :3, 3], rw[-1, :3, 3]
        print(f"LW_T0:   {lt0.tolist()}")
        print(f"LW_LAST: {lt_last.tolist()}")
        print(f"RW_T0:   {rt0.tolist()}")
        print(f"RW_LAST: {rt_last.tolist()}")
        print(f"LW_TRANS_DELTA_MAG: {float(np.linalg.norm(lt_last - lt0)):.4f}")
        print(f"RW_TRANS_DELTA_MAG: {float(np.linalg.norm(rt_last - rt0)):.4f}")
        if "state/raw/robot/root_pos_w" in f:
            root = f["state/raw/robot/root_pos_w"][...]
            travel = float(np.linalg.norm(root[-1] - root[0]))
            print(f"ROOT_TRAVEL (world, m): {travel:.4f}")
            print(
                "(If arms didn't physically move across the rollout, DELTA_MAGs should"
                " be small regardless of ROOT_TRAVEL. Scaling with ROOT_TRAVEL means"
                " world-frame bleed-through.)"
            )


def debug_5_fk_roundtrip(path: Path) -> None:
    """Reconstruct world-frame wrist pose at some step and cross-check vs the
    implied frame from the previous step's root pose — consistency-only."""
    print(f"===== [DEBUG-5] FK ROUND-TRIP (self-consistent, {path.name}) =====")
    with h5py.File(path, "r") as f:
        if "teleop" not in f:
            print("no /teleop group -> skip")
            return
        if "state/raw/robot/root_pos_w" not in f:
            print("state/raw/robot/root_pos_w missing -> skip")
            return
        lw = f["teleop/left_wrist"][...]
        rw = f["teleop/right_wrist"][...]
        root_pos = f["state/raw/robot/root_pos_w"][...]
        root_quat = f["state/raw/robot/root_quat_w"][...]
        N = min(lw.shape[0], root_pos.shape[0])
        if N < 2:
            print("fewer than 2 frames -> skip")
            return

        for step in (0, N // 2, N - 1):
            R_root = _quat_wxyz_to_matrix(root_quat[step])
            T_pelvis_to_world = np.eye(4)
            T_pelvis_to_world[:3, :3] = R_root
            T_pelvis_to_world[:3, 3] = root_pos[step]

            lw_world = T_pelvis_to_world @ lw[step]
            rw_world = T_pelvis_to_world @ rw[step]
            print(
                f"step={step:4d}  "
                f"LW_world_pos={lw_world[:3, 3].round(4).tolist()}  "
                f"RW_world_pos={rw_world[:3, 3].round(4).tolist()}"
            )
        print(
            "(Eyeball the reconstructed world-frame wrist positions. They should sit"
            " ~robot-scale above/around ROOT_POS_W at the same step.)"
        )


# ---------- cross-file checks ----------

def debug_6_finger_layout(paths: list[Path]) -> None:
    print("===== [DEBUG-6] FINGER LAYOUT STABILITY =====")
    left_lists: list[list[str]] = []
    right_lists: list[list[str]] = []
    for p in paths:
        with h5py.File(p, "r") as f:
            if "teleop/finger_joints" not in f:
                print(f"{p.name}: no /teleop/finger_joints -> skip")
                return
            fj = f["teleop/finger_joints"]
            ln = [n.decode() for n in fj["left_finger_joint_names"][...]]
            rn = [n.decode() for n in fj["right_finger_joint_names"][...]]
            print(f"{p.name}")
            print(f"  LEFT_NAMES:  {ln}")
            print(f"  RIGHT_NAMES: {rn}")
            print(f"  LEFT_SHAPE:  {fj['left'].shape}  RIGHT_SHAPE: {fj['right'].shape}")
            left_lists.append(ln)
            right_lists.append(rn)
    all_left_same = all(lst == left_lists[0] for lst in left_lists)
    all_right_same = all(lst == right_lists[0] for lst in right_lists)
    print(f"ALL_LEFT_NAMES_EQUAL:  {all_left_same}")
    print(f"ALL_RIGHT_NAMES_EQUAL: {all_right_same}")


def debug_7_determinism(paths: list[Path]) -> None:
    print("===== [DEBUG-7] DETERMINISM =====")
    hashes: list[tuple[int, int, int]] = []
    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            lw = f["teleop/left_wrist"][...]
            rw = f["teleop/right_wrist"][...]
            h = (hash(lw.tobytes()), hash(rw.tobytes()), int(lw.shape[0]))
            print(f"{i}  {p.name}  lw_hash={h[0]}  rw_hash={h[1]}  N={h[2]}")
            hashes.append(h)
    print(f"ALL_IDENTICAL: {all(h == hashes[0] for h in hashes)}")
    print("(Only meaningful if the two rollouts were produced with identical CLI args + seed.)")


# ---------- entry point ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop HDF5 debug checks.")
    parser.add_argument("--hdf5", nargs="+", required=True, help="One or more rollout HDF5 paths.")
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.hdf5]
    for p in paths:
        if not p.exists():
            print(f"[ERROR] file does not exist: {p}")
            sys.exit(1)

    for p in paths:
        print()
        print("#" * 70)
        print(f"# {p}")
        print("#" * 70)
        debug_1_schema(p)
        debug_2_se3(p)
        debug_3_frame0(p)
        debug_4_locomotion_invariance(p)
        debug_5_fk_roundtrip(p)

    if len(paths) >= 2:
        print()
        print("#" * 70)
        print("# CROSS-FILE CHECKS")
        print("#" * 70)
        debug_6_finger_layout(paths)
        debug_7_determinism(paths)


if __name__ == "__main__":
    main()
