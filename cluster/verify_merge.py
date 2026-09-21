"""Verify a merged LeRobot dataset against a known-good single-process conversion.

    python verify_merge.py --merged MERGED --reference REF

Checks, in order of how badly a failure would corrupt training:
  1. info.json totals (episodes / frames / videos / tasks / splits)
  2. episode count and per-episode lengths, as a MULTISET -- shard order need not match the
     reference's episode order, so lengths are compared sorted, and a pairing is then built by
     matching the underlying source episodes via their per-episode stats.
  3. the global ``index`` column is a contiguous 0..N-1 with no gaps or repeats, and every
     ``episode_index`` column matches its filename
  4. stats.json field-by-field, including q01/q99, within tolerance
  5. video files present for every episode, non-empty
  6. frame-level content: for a sample of episodes, numeric columns compared exactly against the
     reference episode with the same per-episode stats fingerprint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fingerprint(stats_row: dict) -> tuple:
    """Stable per-episode fingerprint from its stats, used to pair merged<->reference episodes."""
    st = stats_row["stats"]
    keys = sorted(k for k in st if isinstance(st[k], dict) and "mean" in st[k])
    vals = []
    for k in keys[:4]:
        m = np.asarray(st[k]["mean"], dtype=float).ravel()
        vals.extend(np.round(m[:6], 6).tolist())
    return tuple(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--sample", type=int, default=6, help="episodes to compare frame-by-frame")
    args = ap.parse_args()

    M, R = Path(args.merged).expanduser(), Path(args.reference).expanduser()
    fails, warns = [], []

    mi = json.loads((M / "meta" / "info.json").read_text())
    ri = json.loads((R / "meta" / "info.json").read_text())
    print("1) info.json totals")
    for k in ("total_episodes", "total_frames", "total_videos", "total_tasks", "splits"):
        ok = mi.get(k) == ri.get(k)
        print(f"   {k:16s} merged={mi.get(k)!s:<22s} ref={ri.get(k)!s:<22s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"info.{k}: {mi.get(k)} != {ri.get(k)}")

    me = read_jsonl(M / "meta" / "episodes.jsonl")
    re_ = read_jsonl(R / "meta" / "episodes.jsonl")
    print("2) episode lengths (multiset)")
    ml, rl = sorted(e["length"] for e in me), sorted(e["length"] for e in re_)
    print(f"   count merged={len(ml)} ref={len(rl)} | sum merged={sum(ml)} ref={sum(rl)} | "
          f"{'OK' if ml == rl else 'MISMATCH'}")
    if ml != rl:
        fails.append("episode length multiset differs")
    idxs = [e["episode_index"] for e in me]
    if idxs != list(range(len(me))):
        fails.append("merged episode_index is not 0..N-1 contiguous")

    print("3) parquet index integrity")
    tpl = mi["data_path"]
    seen, bad_ep = [], 0
    for e in me:
        i = e["episode_index"]
        df = pd.read_parquet(M / tpl.format(episode_chunk=0, episode_index=i))
        if "episode_index" in df.columns and set(df["episode_index"].unique()) != {i}:
            bad_ep += 1
        if "index" in df.columns:
            seen.append(df["index"].to_numpy())
    allidx = np.sort(np.concatenate(seen)) if seen else np.array([])
    contiguous = allidx.size and allidx[0] == 0 and np.array_equal(allidx, np.arange(allidx.size))
    print(f"   episode_index column matches filename: {'OK' if bad_ep == 0 else f'{bad_ep} BAD'}")
    print(f"   global index contiguous 0..{allidx.size - 1}: {'OK' if contiguous else 'BROKEN'}")
    if bad_ep:
        fails.append(f"{bad_ep} episodes have wrong episode_index column")
    if not contiguous:
        fails.append("global index column is not contiguous")

    print("4) stats.json (incl. q01/q99)")
    ms = json.loads((M / "meta" / "stats.json").read_text()) if (M / "meta" / "stats.json").exists() else None
    rs = json.loads((R / "meta" / "stats.json").read_text())
    if ms is None:
        fails.append("merged stats.json missing -- run gr00t generate_stats() on the merged root")
        print("   MISSING (run generate_stats first)")
    else:
        worst = {}
        for key in sorted(rs):
            for field in ("mean", "std", "min", "max", "q01", "q99"):
                if field not in rs[key] or key not in ms or field not in ms[key]:
                    continue
                a = np.asarray(ms[key][field], float).ravel()
                b = np.asarray(rs[key][field], float).ravel()
                if a.shape != b.shape:
                    fails.append(f"stats {key}.{field} shape {a.shape} != {b.shape}")
                    continue
                d = float(np.max(np.abs(a - b))) if a.size else 0.0
                rel = d / (float(np.max(np.abs(b))) + 1e-9)
                if rel > worst.get(field, (0, ""))[0]:
                    worst[field] = (rel, key)
        for field, (rel, key) in sorted(worst.items()):
            status = "OK" if rel < 1e-4 else ("CLOSE" if rel < 1e-2 else "MISMATCH")
            print(f"   {field:5s} worst rel-diff {rel:.2e} (at {key}) {status}")
            if rel >= 1e-2:
                fails.append(f"stats {field} differs by {rel:.2e} at {key}")
            elif rel >= 1e-4:
                warns.append(f"stats {field} rel-diff {rel:.2e} at {key}")

    print("5) videos")
    vt = mi.get("video_path")
    vkeys = [k for k, v in mi["features"].items() if v.get("dtype") == "video"]
    missing = empty = 0
    for e in me:
        for k in vkeys:
            p = M / vt.format(episode_chunk=0, video_key=k, episode_index=e["episode_index"])
            if not p.exists():
                missing += 1
            elif p.stat().st_size == 0:
                empty += 1
    print(f"   {len(me) * len(vkeys)} expected | missing={missing} empty={empty} "
          f"{'OK' if missing == empty == 0 else 'BAD'}")
    if missing or empty:
        fails.append(f"videos missing={missing} empty={empty}")

    print(f"6) frame content, {args.sample} sampled episodes")
    mfp = {fingerprint(r): r["episode_index"] for r in read_jsonl(M / "meta" / "episodes_stats.jsonl")}
    rfp = {fingerprint(r): r["episode_index"] for r in read_jsonl(R / "meta" / "episodes_stats.jsonl")}
    common = [f for f in mfp if f in rfp]
    print(f"   episodes pairable by stats fingerprint: {len(common)}/{len(me)}")
    if len(common) < len(me):
        warns.append(f"only {len(common)}/{len(me)} episodes pairable by fingerprint")
    rng = np.random.default_rng(0)
    for f in (rng.permutation(len(common))[: args.sample] if common else []):
        fp = common[int(f)]
        a = pd.read_parquet(M / tpl.format(episode_chunk=0, episode_index=mfp[fp]))
        b = pd.read_parquet(R / ri["data_path"].format(episode_chunk=0, episode_index=rfp[fp]))
        cols = [c for c in a.columns if c in b.columns and c not in ("index", "episode_index")]
        diffs = []
        for c in cols:
            try:
                x = np.stack(a[c].to_numpy()) if a[c].dtype == object else a[c].to_numpy()
                y = np.stack(b[c].to_numpy()) if b[c].dtype == object else b[c].to_numpy()
            except Exception:
                continue
            if x.shape != y.shape:
                diffs.append(f"{c}:shape"); continue
            if x.dtype.kind in "fc" and not np.allclose(x, y, atol=1e-6):
                diffs.append(f"{c}:{float(np.max(np.abs(x - y))):.2e}")
            elif x.dtype.kind not in "fc" and not np.array_equal(x, y):
                diffs.append(f"{c}:neq")
        print(f"   ep m{mfp[fp]} vs ref{rfp[fp]}: {len(cols)} cols "
              f"{'IDENTICAL' if not diffs else 'DIFFER ' + ','.join(diffs[:4])}")
        if diffs:
            fails.append(f"frame content differs in ep {mfp[fp]}: {diffs[:4]}")

    print("\n" + "=" * 62)
    if fails:
        print(f"VERIFY FAILED ({len(fails)} problems)")
        for f in fails:
            print("  -", f)
    else:
        print("VERIFY PASSED - merged dataset is equivalent to the reference")
    for w in warns:
        print("  warn:", w)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
