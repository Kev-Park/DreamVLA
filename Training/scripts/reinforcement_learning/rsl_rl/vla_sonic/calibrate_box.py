"""Calibrate the 4-rule palm box from an expert's own demos.

    python -m vla_sonic.calibrate_box --out box.json <demo hdf5 dir or glob> [...]

Same procedure as BOX_*_AF60 / BOX_*_AF60V7 in grasp_success.py: p05/p95 of the object's palm-frame
position over the hold phase (the longest contiguous commanded-closed run, >= 25 frames), pooled over the
episodes that pass rules 1/3/4. Each expert seats the bottle differently in the palm (af60v2 ~10 cm
higher than pre-audit, af60v7 ~2.5 cm lower than af60v2), so a box calibrated on one expert can reject
good grasps of another. The JSON feeds collect_sonic_adapter.py --dagger-rescue-box file:<json> and
score(box_lo=, box_hi=) in offline filters / evals.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

from vla_sonic.grasp_success import BOX_HI_AF60, BOX_LO_AF60, score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="dirs (searched recursively) or globs of demo .hdf5 files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-hold", type=int, default=25)
    a = ap.parse_args()
    files: list[str] = []
    for x in a.inputs:
        files += glob.glob(os.path.join(x, "**", "*.hdf5"), recursive=True) if os.path.isdir(x) else glob.glob(x)
    files = sorted(f for f in set(files) if "_passA" not in os.path.basename(f))
    held, n_eps = [], 0
    for f in files:
        with h5py.File(f, "r", locking=False) as h:
            g = h["data/demo_0"]
            args = (g["obs/object_pos"][()], g["obs/object_quat"][()], g["obs/robot0_root_pos_w"][()],
                    g["obs/robot0_root_quat_w"][()], g["teleop/right_wrist"][()])
            fing = g["actions"][()][:, 64]
        r = score(*args, box_lo=BOX_LO_AF60, box_hi=BOX_HI_AF60)       # box-independent rules only are used
        if r["fallen_height"] or r["fallen_horizontal"] or r["slipping"]:
            continue
        c = np.r_[False, fing < 0, False].astype(int)
        d = np.diff(c); s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if not len(s):
            continue
        i = int(np.argmax(e - s))
        if e[i] - s[i] < a.min_hold:
            continue
        held.append(r["rel"][s[i]:e[i]]); n_eps += 1
    if not held:
        raise SystemExit(f"no clean hold phases in {len(files)} files")
    H = np.concatenate(held)
    lo, hi = np.percentile(H, 5, axis=0), np.percentile(H, 95, axis=0)
    out = {"lo": [round(float(v), 3) for v in lo], "hi": [round(float(v), 3) for v in hi],
           "episodes": n_eps, "frames": int(len(H)), "files_scanned": len(files),
           "procedure": "p05/p95 palm-frame object position, longest commanded-closed run, episodes passing rules 1/3/4"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[calibrate_box] {n_eps} clean episodes / {len(H)} frames -> lo {out['lo']} hi {out['hi']} -> {a.out}")


if __name__ == "__main__":
    main()
