"""Verify a merged (sharded) LeRobot conversion against a single-process conversion of the SAME inputs.

    python verify_merge.py --merged MERGED --reference REF

Because convert_sharded.sh shards contiguous slices of the sorted input list and merge_lerobot.py
concatenates them in order, the two datasets should be identical episode for episode, so this checks
for EXACT equality (no fuzzy pairing):
  1. the set of files (stats.json / relative_stats.json excluded: GR00T generates them later)
  2. every video: byte-identical
  3. every parquet: byte-identical; if not, the same Arrow schema + identical values column by column
  4. info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, modality.json: byte-identical
Run gr00t's generate_stats / generate_rel_stats on both roots afterwards and compare those too
(convert_sharded.sh --validate does all of this).
"""

from __future__ import annotations

import argparse
import filecmp
from pathlib import Path

import pyarrow.parquet as pq

GENERATED = {"meta/stats.json", "meta/relative_stats.json"}


def files(root: Path) -> set[str]:
    return {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()} - GENERATED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--reference", required=True)
    a = ap.parse_args()
    M, R = Path(a.merged).expanduser(), Path(a.reference).expanduser()
    fails: list[str] = []

    fm, fr = files(M), files(R)
    print(f"1) files: merged {len(fm)}, reference {len(fr)}")
    if fm != fr:
        fails.append(f"file sets differ: only merged {sorted(fm - fr)[:5]}, only reference {sorted(fr - fm)[:5]}")

    common = sorted(fm & fr)
    vids = [f for f in common if f.endswith(".mp4")]
    bad_v = [f for f in vids if not filecmp.cmp(M / f, R / f, shallow=False)]
    print(f"2) videos byte-identical: {len(vids) - len(bad_v)}/{len(vids)}")
    fails += [f"video differs: {f}" for f in bad_v[:5]]

    pqs = [f for f in common if f.endswith(".parquet")]
    byte_eq = value_eq = 0
    for f in pqs:
        if filecmp.cmp(M / f, R / f, shallow=False):
            byte_eq += 1
            continue
        tm, tr = pq.read_table(M / f), pq.read_table(R / f)
        if tm.schema.equals(tr.schema, check_metadata=True) and tm.equals(tr):
            value_eq += 1
        else:
            cols = [c for c in tr.column_names if c not in tm.column_names or not tm[c].equals(tr[c])]
            fails.append(f"parquet differs: {f} (schema_eq={tm.schema.equals(tr.schema)}, columns {cols[:4]})")
    print(f"3) parquet: byte-identical {byte_eq}, value+schema-identical {value_eq}, of {len(pqs)}")

    metas = [f for f in common if f.startswith("meta/")]
    bad_m = [f for f in metas if not filecmp.cmp(M / f, R / f, shallow=False)]
    print(f"4) meta files byte-identical: {len(metas) - len(bad_m)}/{len(metas)} {bad_m if bad_m else ''}")
    fails += [f"meta differs: {f}" for f in bad_m]

    print("\n" + ("VERIFY PASSED - merged dataset is identical to the single-process conversion"
                  if not fails else f"VERIFY FAILED ({len(fails)} problems)"))
    for f in fails:
        print("  -", f)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
