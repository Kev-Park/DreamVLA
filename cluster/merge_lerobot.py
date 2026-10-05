"""Merge LeRobot v2.1 shard datasets into the dataset a single-process conversion would have written.

    python merge_lerobot.py --shards SHARD0 SHARD1 ... --out MERGED

The shards must come from convert_isaac_hdf5_to_lerobot.py run over CONTIGUOUS slices of the same
sorted input list, given here in input order (convert_sharded.sh does exactly that). The converter
sorts its inputs globally, so concatenating such shards reproduces the single-process episode order,
and the only thing that differs is the numbering -- every shard starts at episode 0 / frame 0. Per
episode this rewrites:
  * the parquet's ``episode_index`` and global ``index`` columns, written back with the same
    ``datasets.Dataset.to_parquet`` writer LeRobot uses (same schema + embedded HF features);
  * the video's file name (copied byte for byte, never re-encoded);
  * its episodes.jsonl / episodes_stats.jsonl lines -- the stats of ``episode_index`` and ``index``
    are recomputed with LeRobot's own ``compute_episode_stats`` on the offset arrays, built exactly
    as the exporter builds them, so they match bit for bit;
and writes info.json with the merged totals via LeRobot's ``write_json``.

stats.json and relative_stats.json are deliberately NOT written (and never copied from a shard):
the converter does not write them; GR00T's generate_stats()/generate_rel_stats() compute them from
the parquet at fine-tune time over the WHOLE dataset. A shard's copy would carry that shard's
q01/q99 and be accepted as valid, silently mis-normalising training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import datasets
import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.utils import serialize_dict, write_json


def _jsonl(p: Path) -> list[dict]:
    with jsonlines.open(p) as r:
        return list(r)


def merge(shards: list[Path], out: Path) -> None:
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in shards]
    base = infos[0]
    for s, inf in zip(shards, infos):
        for k in ("features", "fps", "codebase_version", "robot_type", "chunks_size", "script_config",
                  "data_path", "video_path"):
            if inf.get(k) != base.get(k):
                sys.exit(f"{s}: info['{k}'] differs from shard 0 -- not shards of one conversion")
        if inf.get("discarded_episode_indices"):
            sys.exit(f"{s}: has discarded episodes; merging those is not supported")
        if (s / "meta" / "tasks.jsonl").read_bytes() != (shards[0] / "meta" / "tasks.jsonl").read_bytes():
            sys.exit(f"{s}: tasks.jsonl differs from shard 0")

    feats, chunks_size = base["features"], base["chunks_size"]
    data_tpl, video_tpl = base["data_path"], base.get("video_path")
    video_keys = [k for k, v in feats.items() if v["dtype"] == "video"]
    stat_feats = {k: {**feats[k], "shape": tuple(feats[k]["shape"])} for k in ("episode_index", "index")}

    (out / "meta").mkdir(parents=True)
    shutil.copy2(shards[0] / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")
    if (shards[0] / "meta" / "modality.json").exists():
        shutil.copy2(shards[0] / "meta" / "modality.json", out / "meta" / "modality.json")
    ep_w = jsonlines.open(out / "meta" / "episodes.jsonl", "w")
    st_w = jsonlines.open(out / "meta" / "episodes_stats.jsonl", "w")

    g_ep = g_frame = 0
    for s in shards:
        eps = sorted(_jsonl(s / "meta" / "episodes.jsonl"), key=lambda e: e["episode_index"])
        sts = {e["episode_index"]: e["stats"] for e in _jsonl(s / "meta" / "episodes_stats.jsonl")}
        for e in eps:
            li, n = e["episode_index"], e["length"]
            src = s / data_tpl.format(episode_chunk=li // chunks_size, episode_index=li)
            dst = out / data_tpl.format(episode_chunk=g_ep // chunks_size, episode_index=g_ep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tbl = pq.read_table(src)
            if tbl.num_rows != n:
                sys.exit(f"{src}: {tbl.num_rows} rows but episodes.jsonl says {n}")
            ep_arr = np.full((n,), g_ep)                      # exactly as the exporter builds them
            ix_arr = np.arange(g_frame, g_frame + n)
            for col, arr in (("episode_index", ep_arr), ("index", ix_arr)):
                i = tbl.schema.get_field_index(col)
                f = tbl.schema.field(i)
                tbl = tbl.set_column(i, f, pa.array(arr, type=f.type))
            datasets.Dataset(tbl).to_parquet(str(dst))
            for vk in video_keys:
                vs = s / video_tpl.format(episode_chunk=li // chunks_size, video_key=vk, episode_index=li)
                vd = out / video_tpl.format(episode_chunk=g_ep // chunks_size, video_key=vk, episode_index=g_ep)
                vd.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vs, vd)
            ep_w.write({"episode_index": g_ep, "tasks": e["tasks"], "length": n})
            fresh = serialize_dict(compute_episode_stats({"episode_index": ep_arr, "index": ix_arr}, stat_feats))
            st_w.write({"episode_index": g_ep, "stats": {k: fresh.get(k, v) for k, v in sts[li].items()}})
            g_ep += 1
            g_frame += n
    ep_w.close()
    st_w.close()

    info = dict(base)
    info.update(total_episodes=g_ep, total_frames=g_frame, total_videos=g_ep * len(video_keys),
                total_chunks=(g_ep - 1) // chunks_size + 1 if g_ep else 0,
                splits={"train": f"0:{g_ep}"}, discarded_episode_indices=[])
    write_json(info, out / "meta" / "info.json")
    print(f"[merge] {len(shards)} shards -> {g_ep} episodes / {g_frame} frames -> {out}")
    print("[merge] stats.json / relative_stats.json intentionally absent: GR00T generates them at fine-tune time")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="shard dataset roots, in input order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge([Path(s).expanduser() for s in a.shards], Path(a.out).expanduser())
