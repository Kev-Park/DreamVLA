"""Merge N shard LeRobot datasets (v2.1) into one.

    python merge_lerobot.py --shards SHARD0 SHARD1 ... --out MERGED [--name NAME]

Why merging is not just renaming files:
  * every episode parquet carries an ``episode_index`` column and a GLOBAL ``index`` column;
    both must be rewritten, or the merged dataset silently mislabels frames.
  * ``meta/stats.json`` carries q01/q99, which CANNOT be aggregated from ``episodes_stats.jsonl``
    (those hold only count/max/mean/min/std). GR00T normalises on exactly those percentiles, so
    this script deliberately does NOT write stats.json -- run gr00t's own ``generate_stats()`` on
    the merged root afterwards, which recomputes them from the merged parquet the same way a
    single-process conversion would (see gr00t/data/stats.py:92).

Everything else -- episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, info.json totals -- is a
concatenate-and-renumber.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard dataset roots, in order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards = [Path(s).expanduser() for s in args.shards]
    out = Path(args.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)

    base_info = json.loads((shards[0] / "meta" / "info.json").read_text())
    data_tpl = base_info["data_path"]          # e.g. data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    video_tpl = base_info.get("video_path")
    video_keys = [k for k, v in base_info["features"].items() if v.get("dtype") == "video"]

    ep_rows: list[dict] = []
    stat_rows: list[dict] = []
    tasks: list[dict] = []
    task_seen: dict[str, int] = {}
    new_idx = 0
    global_index = 0
    total_frames = 0

    for sh in shards:
        s_eps = {r["episode_index"]: r for r in read_jsonl(sh / "meta" / "episodes.jsonl")}
        s_stats = {r["episode_index"]: r for r in read_jsonl(sh / "meta" / "episodes_stats.jsonl")}
        for t in read_jsonl(sh / "meta" / "tasks.jsonl"):
            if t["task"] not in task_seen:
                task_seen[t["task"]] = len(tasks)
                tasks.append({"task_index": len(tasks), "task": t["task"]})

        for old_idx in sorted(s_eps):
            src = sh / data_tpl.format(episode_chunk=0, episode_index=old_idx)
            df = pd.read_parquet(src)
            n = len(df)
            df["episode_index"] = new_idx
            df["index"] = range(global_index, global_index + n)     # GLOBAL frame index
            if "task_index" in df.columns:
                # single-task datasets here, but remap defensively via the episode's task name
                tname = s_eps[old_idx]["tasks"][0]
                df["task_index"] = task_seen[tname]
            dst = out / data_tpl.format(episode_chunk=0, episode_index=new_idx)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)

            for key in video_keys:
                vs = sh / video_tpl.format(episode_chunk=0, video_key=key, episode_index=old_idx)
                vd = out / video_tpl.format(episode_chunk=0, video_key=key, episode_index=new_idx)
                vd.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vs, vd)

            row = dict(s_eps[old_idx]); row["episode_index"] = new_idx
            ep_rows.append(row)
            if old_idx in s_stats:
                srow = dict(s_stats[old_idx]); srow["episode_index"] = new_idx
                stat_rows.append(srow)

            new_idx += 1
            global_index += n
            total_frames += n

    write_jsonl(out / "meta" / "episodes.jsonl", ep_rows)
    write_jsonl(out / "meta" / "episodes_stats.jsonl", stat_rows)
    write_jsonl(out / "meta" / "tasks.jsonl", tasks)
    for extra in ("modality.json", "relative_stats.json"):
        src = shards[0] / "meta" / extra
        if src.exists():
            shutil.copy2(src, out / "meta" / extra)

    info = dict(base_info)
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_videos"] = new_idx * max(len(video_keys), 1)
    info["total_tasks"] = len(tasks)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{new_idx}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"[merge] {len(shards)} shards -> {new_idx} episodes, {total_frames} frames -> {out}")
    print("[merge] stats.json intentionally NOT written -- run gr00t generate_stats() on this root")


if __name__ == "__main__":
    main()
