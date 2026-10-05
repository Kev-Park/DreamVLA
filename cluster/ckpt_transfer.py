#!/usr/bin/env python3
"""Ship a GR00T fine-tune checkpoint between clusters without its frozen backbone.

A GR00T fine-tune with tune_llm/tune_visual off changes only `action_head.*` (6.0 of 11.7 GiB);
the `backbone.*` tensors are bit-identical to every other fine-tune of the same base model. So:

  pack    on the training cluster: write the trained tensors + the small config files + a manifest
          holding the sha256 of EVERY tensor (packed or not) and the original shard layout.
  unpack  on the receiving cluster: rebuild the original shards from the pack plus the backbone of
          any local checkpoint of the same base model, verifying every tensor against the manifest.
          A backbone mismatch (e.g. the run actually trained the backbone) is a hard error, so a
          rebuilt checkpoint is either bit-identical to the original or not produced at all.

  python ckpt_transfer.py pack   <checkpoint-dir> <pack-dir>   [--prefix action_head.]
  python ckpt_transfer.py unpack <pack-dir> <base-checkpoint-dir> <out-checkpoint-dir>

The fine-tune process is untouched; this only reads and writes checkpoint directories.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.numpy import save_file

INDEX = "model.safetensors.index.json"
PACKED = "trained.safetensors"
MANIFEST = "manifest.json"


def _sha(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _open_shards(ckpt):
    index = json.load(open(os.path.join(ckpt, INDEX)))
    handles = {}

    def get(key):
        shard = index["weight_map"][key]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(ckpt, shard), framework="numpy")
        return handles[shard].get_tensor(key)

    return index, get, handles


def pack(ckpt, out, prefixes):
    index, get, handles = _open_shards(ckpt)
    os.makedirs(out, exist_ok=True)
    packed, hashes = {}, {}
    for key in sorted(index["weight_map"]):
        t = get(key)
        hashes[key] = _sha(t)
        if key.startswith(tuple(prefixes)):
            packed[key] = t
    if not packed:
        sys.exit(f"no tensors match {prefixes}")
    shard_meta = {s: (h.metadata() or {}) for s, h in handles.items()}
    save_file(packed, os.path.join(out, PACKED), metadata={"format": "pt"})
    for name in os.listdir(ckpt):                      # configs, statistics, trainer state ...
        src = os.path.join(ckpt, name)
        if name.endswith(".safetensors") or name == INDEX:
            continue
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(out, name))
    json.dump({"index": index, "sha256": hashes, "packed_keys": sorted(packed),
               "shard_metadata": shard_meta, "prefixes": prefixes},
              open(os.path.join(out, MANIFEST), "w"))
    gib = sum(t.nbytes for t in packed.values()) / 1024**3
    print(f"packed {len(packed)}/{len(hashes)} tensors ({gib:.2f} GiB) -> {out}")


def unpack(pack_dir, base, out):
    man = json.load(open(os.path.join(pack_dir, MANIFEST)))
    index, hashes, packed_keys = man["index"], man["sha256"], set(man["packed_keys"])
    trained = safe_open(os.path.join(pack_dir, PACKED), framework="numpy")
    base_index, base_get, _ = _open_shards(base)
    missing = set(index["weight_map"]) - packed_keys - set(base_index["weight_map"])
    if missing:
        sys.exit(f"base checkpoint lacks {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    os.makedirs(out, exist_ok=True)
    shards = {}
    for key, shard in index["weight_map"].items():
        shards.setdefault(shard, []).append(key)
    bad = []
    for shard, keys in sorted(shards.items()):
        tensors = {}
        for key in keys:
            t = trained.get_tensor(key) if key in packed_keys else base_get(key)
            if _sha(t) != hashes[key]:
                bad.append(key)
            tensors[key] = t
        if bad:
            sys.exit(f"sha256 mismatch on {len(bad)} tensors (e.g. {bad[:3]}): the base checkpoint's "
                     f"frozen weights differ from the original's -- refusing to write a checkpoint")
        save_file(tensors, os.path.join(out, shard), metadata=man["shard_metadata"].get(shard) or {"format": "pt"})
    json.dump(index, open(os.path.join(out, INDEX), "w"), indent=2)
    for name in os.listdir(pack_dir):
        src = os.path.join(pack_dir, name)
        if name in (PACKED, MANIFEST):
            continue
        dst = os.path.join(out, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"rebuilt {len(hashes)} tensors in {len(shards)} shards, all sha256-verified -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("ckpt"); p.add_argument("out")
    p.add_argument("--prefix", action="append", default=None, help="tensor-name prefix to ship (repeatable)")
    u = sub.add_parser("unpack"); u.add_argument("pack"); u.add_argument("base"); u.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.ckpt, a.out, a.prefix or ["action_head."])
    else:
        unpack(a.pack, a.base, a.out)
