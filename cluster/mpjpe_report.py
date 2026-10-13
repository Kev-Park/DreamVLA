"""Report paired-reference body MPJPE from eval_vla_sonic trajectory dumps.

Root alignment removes translation only, with no per-frame rotation or scale fitting.
Both metrics use the existing 14 HOI body landmarks, including the pelvis.
"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np


def measure(body, reference, root, reference_root):
    body, reference, root, reference_root = map(np.asarray, (body, reference, root, reference_root))
    if body.shape != reference.shape or body.ndim != 3 or body.shape[-1] != 3:
        raise ValueError("Expected matching [frames, bodies, 3] positions")
    if root.shape != (len(body), 3) or reference_root.shape != root.shape or not len(body):
        raise ValueError("Expected nonempty frames and matching root positions")
    absolute = np.linalg.norm(body - reference, axis=-1)
    aligned = np.linalg.norm((body - root[:, None]) - (reference - reference_root[:, None]), axis=-1)
    if not np.isfinite(absolute).all() or not np.isfinite(aligned).all():
        raise ValueError("Nonfinite positions")
    return {"frames": len(body), "bodies": body.shape[1],
            "world_mpjpe_cm": float(absolute.mean() * 100),
            "root_aligned_mpjpe_cm": float(aligned.mean() * 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episodes, missing = [], []
    names = None
    for filename in sorted(glob.glob(str(Path(args.pattern).expanduser()))):
        with np.load(filename, allow_pickle=False) as z:
            required = {"body_pos", "ref_body_pos", "root_pos", "ref_root_pos", "body_names"}
            if not required.issubset(z.files):
                missing.append(filename)
                continue
            current = z["body_names"].tolist()
            if names is not None and current != names:
                raise ValueError("Cannot aggregate different body mappings")
            names = current
            row = measure(z["body_pos"], z["ref_body_pos"], z["root_pos"], z["ref_root_pos"])
            episodes.append(dict(file=filename, **row))
    report = {"body_names": names, "episodes": episodes, "missing_metric_data": missing,
              "alignment": "root translation only; no rotation or scale fitting"}
    if episodes:
        for metric in ("world_mpjpe_cm", "root_aligned_mpjpe_cm"):
            report[metric + "_episode_mean"] = float(np.mean([r[metric] for r in episodes]))
            report[metric + "_frame_mean"] = float(np.average([r[metric] for r in episodes], weights=[r["frames"] for r in episodes]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not episodes:
        raise SystemExit("No trajectories contain MPJPE data")


if __name__ == "__main__":
    main()
