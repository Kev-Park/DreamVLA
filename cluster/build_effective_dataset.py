"""Build balanced episode views of an existing LeRobot v2.1 dataset without copying videos.

This does not generate new trajectories. GR00T applies its existing training image
augmentation when these views are sampled. Original state/action columns are unchanged.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def schedule(indices, count=1200, seed=42):
    if not indices or count < len(indices):
        raise ValueError("Effective dataset must include every source episode")
    rng = random.Random(seed)
    quotient, remainder = divmod(count, len(indices))
    result = list(indices) * quotient + rng.sample(list(indices), remainder)
    rng.shuffle(result)
    return result


def index_stats(values):
    return {"min": [int(values.min())], "max": [int(values.max())],
            "mean": [float(values.mean())], "std": [float(values.std())], "count": [len(values)]}


def validate(source, out):
    info = json.loads((out / "meta/info.json").read_text())
    source_info = json.loads((source / "meta/info.json").read_text())
    manifest = json.loads((out / "meta/effective_dataset.json").read_text())
    episodes = read_jsonl(out / "meta/episodes.jsonl")
    if len(episodes) != manifest["effective_episodes"] or info["total_episodes"] != len(episodes):
        raise ValueError("Episode totals do not match")
    chunk = info["chunks_size"]
    cache = {}
    frame = 0
    for row, mapping in zip(episodes, manifest["episode_mapping"], strict=True):
        dst_id, src_id = row["episode_index"], mapping["source_episode_index"]
        src_file = source / info["data_path"].format(episode_chunk=src_id // chunk, episode_index=src_id)
        if src_id not in cache:
            cache[src_id] = pq.read_table(src_file)
        original = cache[src_id]
        table = pq.read_table(out / info["data_path"].format(episode_chunk=dst_id // chunk, episode_index=dst_id))
        if table.num_rows != row["length"] or not table.schema.equals(original.schema, check_metadata=True):
            raise ValueError("Parquet shape/schema mismatch")
        for name in table.column_names:
            if name not in ("episode_index", "index") and not table[name].equals(original[name]):
                raise ValueError("Original training column changed: " + name)
        if not np.array_equal(table["episode_index"].to_numpy(), np.full(row["length"],dst_id)):
            raise ValueError("Episode indexing mismatch")
        if not np.array_equal(table["index"].to_numpy(), np.arange(frame, frame+row["length"])):
            raise ValueError("Frame indexing mismatch")
        for key, feature in info["features"].items():
            if feature["dtype"] != "video": continue
            src_video=source / info["video_path"].format(episode_chunk=src_id//chunk,video_key=key,episode_index=src_id)
            dst_video=out / info["video_path"].format(episode_chunk=dst_id//chunk,video_key=key,episode_index=dst_id)
            if not dst_video.samefile(src_video) or dst_video.stat().st_size == 0:
                raise ValueError("Video link mismatch")
        frame += row["length"]
    if frame != info["total_frames"] or info["features"] != source_info["features"]:
        raise ValueError("Frame totals or features changed")
    if (out / "meta/stats.json").exists() or (out / "meta/relative_stats.json").exists():
        raise ValueError("Stale normalization statistics must not be copied")
    return {"episodes":len(episodes),"unique_source_episodes":len(cache),"frames":frame,
            "all_original_columns_preserved":True,"all_videos_hardlinked":True}


def build(source, out, count=1200, seed=42):
    source, out = Path(source).expanduser().resolve(), Path(out).expanduser().resolve()
    if out.exists():
        raise FileExistsError("Refusing to overwrite " + str(out))
    temporary = out.with_name(out.name + ".building")
    temporary.mkdir(parents=True, exist_ok=False)
    meta = temporary / "meta"
    meta.mkdir()
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info.get("discarded_episode_indices"):
        raise ValueError("Expected a complete LeRobot v2.1 source dataset")
    episodes = {e["episode_index"]:e for e in read_jsonl(source / "meta/episodes.jsonl")}
    stats = {e["episode_index"]:e["stats"] for e in read_jsonl(source / "meta/episodes_stats.jsonl")}
    ids = schedule(sorted(episodes), count, seed)
    chunk = info["chunks_size"]
    video_keys=[k for k,v in info["features"].items() if v["dtype"]=="video"]
    for name in ("tasks.jsonl","modality.json"):
        if (source / "meta" / name).exists():
            (meta / name).write_bytes((source / "meta" / name).read_bytes())
    cache, mappings, frame = {}, [], 0
    with (meta/"episodes.jsonl").open("w") as ep_out, (meta/"episodes_stats.jsonl").open("w") as stats_out:
        for dst_id, src_id in enumerate(ids):
            episode=episodes[src_id]; n=episode["length"]
            if src_id not in cache:
                cache[src_id]=pq.read_table(source / info["data_path"].format(episode_chunk=src_id//chunk,episode_index=src_id))
            table=cache[src_id]
            ep_arr=np.full(n,dst_id,dtype=np.int64); ix_arr=np.arange(frame,frame+n,dtype=np.int64)
            for name, values in (("episode_index",ep_arr),("index",ix_arr)):
                pos=table.schema.get_field_index(name); field=table.schema.field(pos)
                table=table.set_column(pos,field,pa.array(values,type=field.type))
            dst_file=temporary / info["data_path"].format(episode_chunk=dst_id//chunk,episode_index=dst_id)
            dst_file.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(table,dst_file)
            for key in video_keys:
                src_video=source / info["video_path"].format(episode_chunk=src_id//chunk,video_key=key,episode_index=src_id)
                dst_video=temporary / info["video_path"].format(episode_chunk=dst_id//chunk,video_key=key,episode_index=dst_id)
                dst_video.parent.mkdir(parents=True,exist_ok=True)
                os.link(src_video,dst_video)
            ep_out.write(json.dumps(dict(episode_index=dst_id,tasks=episode["tasks"],length=n))+"\n")
            new_stats=copy.deepcopy(stats[src_id]); new_stats.update(episode_index=index_stats(ep_arr),index=index_stats(ix_arr))
            stats_out.write(json.dumps(dict(episode_index=dst_id,stats=new_stats))+"\n")
            mappings.append(dict(episode_index=dst_id,source_episode_index=src_id))
            frame+=n
    updated=dict(info,total_episodes=count,total_frames=frame,total_videos=count*len(video_keys),
                 total_chunks=math.ceil(count/chunk),splits={"train":f"0:{count}"},discarded_episode_indices=[])
    (meta/"info.json").write_text(json.dumps(updated,indent=2)+"\n")
    manifest=dict(source_dataset=str(source),source_info_sha256=hashlib.sha256((source/"meta/info.json").read_bytes()).hexdigest(),
                  seed=seed,unique_source_episodes=len(episodes),effective_episodes=count,unique_new_trajectories=0,
                  repetitions=dict(sorted(Counter(ids).items())),episode_mapping=mappings,
                  training_augmentation="Existing GR00T training random crop; no added augmentation flags",
                  normalization="Generate over this complete dataset at training time",
                  effective_epochs=5.7,recommended_max_steps=math.ceil(frame*5.7/16),effective_batch_size=16)
    (meta/"effective_dataset.json").write_text(json.dumps(manifest,indent=2)+"\n")
    result=validate(source,temporary)
    (meta/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    (temporary/"READY").write_text("Validated effective dataset; repeated source demos, zero new trajectories.\n")
    temporary.rename(out)
    print(json.dumps(dict(dataset=str(out),**result)),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path);parser.add_argument("out",type=Path)
    parser.add_argument("--count",type=int,default=1200);parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args();build(args.source,args.out,args.count,args.seed)
