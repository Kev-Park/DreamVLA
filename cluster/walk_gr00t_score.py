"""Score exactly the 100 walking holdout references using the pick protocol."""
import json
from pathlib import Path
import re
import sys

import numpy as np
from mpjpe_report import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Training/scripts/reinforcement_learning/rsl_rl"))
from vla_sonic.grasp_success import score


def main():
    out, boxfile = map(Path, sys.argv[1:])
    box = json.loads(boxfile.read_text())
    paths = sorted(out.glob("b*_traj.npz"))
    source_ids = [int(re.search(r"_m(\d+)_traj.npz$", p.name)[1]) for p in paths]
    mapping_file = out/"reference_mapping.json"
    mapping = json.loads(mapping_file.read_text()) if mapping_file.exists() else None
    if mapping:
        assert mapping['pool'] == 'StitchMAX100v3_eval'
        entries = mapping['references']
        assert sorted(r['source_id'] for r in entries) == list(range(100))
        assert sorted(r['canonical_id'] for r in entries) == list(range(100))
        assert len({r['name'] for r in entries}) == 100
        assert all(r['source_sha256'] == r['canonical_sha256'] for r in entries)
        lookup = {r['source_id']: r['canonical_id'] for r in entries}
        ids = [lookup[mid] for mid in source_ids]
    else:
        ids = source_ids
    assert sorted(ids) == list(range(100)), f"Missing/duplicated holdout references: {ids}"
    rows = []
    for path, mid in zip(paths, ids):
        with np.load(path, allow_pickle=False) as z:
            result = score(z["obj_pos"], z["obj_quat"], z["root_pos"], z["root_quat"], z["wrist"],
                           box_lo=np.array(box["lo"]), box_hi=np.array(box["hi"]))
            metrics = measure(z["body_pos"], z["ref_body_pos"], z["root_pos"], z["ref_root_pos"])
            rows.append(dict(motion_id=mid, success=bool(result["success"]), **metrics))
    report = {"pool": "StitchMAX100v3_eval", "episodes": 100,
              "successes": sum(r["success"] for r in rows), "criterion": "calibrated four-rule",
              "world_mpjpe_cm": float(np.mean([r["world_mpjpe_cm"] for r in rows])),
              "root_aligned_mpjpe_cm": float(np.mean([r["root_aligned_mpjpe_cm"] for r in rows])),
              "aggregation": "episode mean; root translation only", "per_episode": rows}
    report["success_pct"] = float(report["successes"])
    if mapping:
        report['reference_mapping'] = mapping
    # Preserve the physical 5cm hold metric as well, so it can be compared with
    # the residual/frozen SONIC evaluations without conflating the criteria.
    totals = []
    for log in sorted(boxfile.parent.glob(f"eval_{out.name}_shard*.log")):
        text = log.read_text(errors="replace")
        match = re.search(r"\[PHYS\] held \(lift>=5cm[^\n]*?:\s*(\d+)\s*/\s*(\d+)", text)
        assert match and "VLA + SONIC EVAL SUMMARY" in text, f"Incomplete eval {log}"
        totals.append(tuple(map(int, match.group(1, 2))))
    assert len(totals) == 4 and sum(n for k, n in totals) == 100
    report["physical_held5_successes"] = sum(k for k, n in totals)
    report["physical_held5_pct"] = float(report["physical_held5_successes"])
    (out/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "per_episode"}))


if __name__ == "__main__":
    main()
