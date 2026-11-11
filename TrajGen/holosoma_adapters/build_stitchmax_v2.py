"""Extend the existing stitch pool using its unchanged stitch recipe and body-pose gate.

StitchMAX200v2 contains 157 relative symlinks plus 43 new original references.
StitchMAX200v2_aug contains one stitched version of each existing static augmentation.
StitchMAX400v2 links those two pools. No input is modified or removed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import time

import numpy as np
import torch
import pytorch_kinematics as pk
import stitch_static_approach as stitcher

ROOT = Path.home() / "kevin" / "ref_motions"
REPORT = Path.home() / "kevin" / "eval_videos" / "StitchMAX200v2"
SHIFT = [0.0332, 0.0281, -0.0394]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def chain():
    torch.set_num_threads(1)
    c = pk.build_chain_from_urdf(Path(stitcher.URDF).read_bytes()).to(dtype=torch.float32)
    return c, [np.asarray(x, dtype=np.float64) for x in c.get_joint_limits()]


def gate(w, s):
    dw, yw = stitcher.body_offset(w, stitcher.final_footfall(w), w["g"])
    ds, ys = stitcher.body_offset(s, s["g"], s["g"])
    pos = float(np.linalg.norm(dw - ds))
    angle = float(abs(np.angle(np.exp(1j * (yw - ys)))))
    return pos, angle, pos <= stitcher.MAX_POS_ERR and angle <= stitcher.MAX_YAW_ERR


def plan():
    c, _ = chain()
    static = ROOT / "Holosoma_Pick_29_latband60max"
    originals = sorted(p for p in static.glob("pick_*.pkl") if re.fullmatch(r"pick_\d+", p.stem))
    augmented = sorted(p for p in static.glob("pick_*.pkl") if p not in originals)
    replacements = list((ROOT / "StitchMAX200v2_static_replacements").glob("pick_*.pkl"))
    for replacement in replacements:
        parent = "_".join(replacement.stem.split("_")[:2])
        matches = [p for p in augmented if "_".join(p.stem.split("_")[:2]) == parent]
        assert len(matches) == 1, parent
        augmented[augmented.index(matches[0])] = replacement
    assert len(originals) == len(augmented) == 200
    original_ids = {p.stem for p in originals}
    assert {"_".join(p.stem.split("_")[:2]) for p in augmented} == original_ids
    old = ROOT / "StitchMAX200"
    existing = sorted(old.glob("pick_*.pkl"))
    assert len(existing) == 157 and {p.stem for p in existing} <= original_ids
    walks = {}
    by_hash = {}
    for p in sorted((ROOT / "WalkMAX").glob("pick_*.pkl")):
        h = sha(p)
        if h not in by_hash:
            by_hash[h] = str(p)
            walks[str(p)] = stitcher.load(str(p), c)
    tasks, failures, links, suggestions = [], [], [], []
    pairings = {}
    for p in originals + augmented:
        is_original = p in originals
        parent = "_".join(p.stem.split("_")[:2])
        preferred = ROOT / "WalkMAX" / (parent + ".pkl")
        s = stitcher.load(str(p), c)
        original_walk = str(preferred)
        if preferred.exists():
            original_walk = by_hash[sha(preferred)]
        if not is_original:
            original_walk = pairings.get(parent, original_walk)
        # Preserve the original assigned walk when it passes; otherwise pick the closest
        # valid existing walk, deterministically. Never widen the original acceptance gate.
        ranked = []
        for wp, w in walks.items():
            pos, angle, ok = gate(w, s)
            if ok:
                ranked.append((wp != original_walk, pos / stitcher.MAX_POS_ERR + angle / stitcher.MAX_YAW_ERR, wp, pos, angle))
        if not ranked:
            failures.append(str(p))
            if not is_original:
                # Suggest a seeded resample within the original augmentation box and
                # lateral band. The existing augmentation CLI performs the actual IK.
                base = stitcher.load(str(static / (parent + ".pkl")), c)
                yaw = stitcher.yaw_of(base["Q"][base["g"]])
                forward = np.array([np.cos(yaw), np.sin(yaw)])
                left = np.array([-np.sin(yaw), np.cos(yaw)])
                rng = np.random.default_rng([1, int(parent.split("_")[1])])
                for attempt in range(50):
                    fwd, lat = float(rng.uniform(-0.04, 0.04)), float(rng.uniform(-0.05, 0.05))
                    trial = dict(base, O=base["O"].copy())
                    trial["O"][:, :2] += fwd * forward + lat * left
                    offset = float((trial["O"][trial["g"], :2] - trial["P"][trial["g"], :2]) @ left)
                    if -0.29 <= offset <= -0.14 and any(gate(w, trial)[2] for w in walks.values()):
                        suggestions.append({"parent": parent, "seed": 1, "draw_index": attempt, "forward_m": fwd, "left_m": lat, "replaces": str(p)})
                        break
            continue
        _, _, wp, pos, angle = min(ranked)
        if is_original and (old / p.name).exists():
            assert wp == original_walk, (p.name, "existing reference does not pass original gate")
            links.append({"name": p.name, "source": str(old / p.name), "sha256": sha(old / p.name)})
            pairings[parent] = wp
            continue
        output = ROOT / ("StitchMAX200v2" if is_original else "StitchMAX200v2_aug") / p.name
        tasks.append({"name": p.name, "original": is_original, "static": str(p), "static_sha256": sha(p), "walk": wp, "walk_sha256": sha(wp), "output": str(output), "repaired_pairing": wp != original_walk, "body_position_error_m": pos, "body_yaw_error_deg": float(np.degrees(angle))})
        if is_original:
            pairings[parent] = wp
    result = {"created": time.time(), "recipe": "existing stitch_static_approach; unchanged 10cm/15deg body gate, full-ramp uniform hand shift", "hand_shift_m": SHIFT, "walk_candidates": len(walks), "links": links, "tasks": tasks, "failures": failures, "resample_suggestions": suggestions, "expected_counts": {"original": 200, "augmented": 200, "training": 400}}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / "plan.json", result)
    print(json.dumps({"links": len(links), "new_originals": sum(t["original"] for t in tasks), "augmentations": sum(not t["original"] for t in tasks), "walk_candidates": len(walks), "failures": failures, "resample_suggestions": suggestions}), flush=True)
    return result


def generate(task):
    c, lim = chain()
    stitcher.HAND_SHIFT = np.asarray(SHIFT)
    stitcher.SHIFT_RAMP_FRAMES = None
    assert sha(task["static"]) == task["static_sha256"]
    assert sha(task["walk"]) == task["walk_sha256"]
    pkl, info = stitcher.stitch(stitcher.load(task["walk"], c), stitcher.load(task["static"], c), c, lim)
    assert pkl is not None, task
    joints = np.asarray(pkl["joints"])
    poses = np.asarray(pkl["object_poses"])
    assert joints.ndim == 2 and joints.shape[1] == 29
    assert len(joints) == len(poses) and 0 < pkl["grab_idx"] < len(joints)
    assert np.isfinite(joints).all() and np.isfinite(poses).all()
    assert np.isfinite(np.asarray(pkl["global_position"])).all()
    assert all(np.isfinite(v) for v in info.values())
    dest = Path(task["output"])
    assert not dest.exists() and not dest.is_symlink(), str(dest)
    tmp = dest.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(pkl, f)
    tmp.replace(dest)
    return {**task, "sha256": sha(dest), "frames": len(joints), "diagnostics": info}


def link(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.is_symlink():
        assert destination.resolve(strict=True) == source.resolve(strict=True)
        return
    assert not destination.exists(), str(destination)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def build():
    p = json.loads((REPORT / "plan.json").read_text())
    assert not p["failures"], "Some source clips have no valid pairing; review plan.json"
    assert len(p["links"]) == 157 and len(p["tasks"]) == 243
    for name in ["StitchMAX200v2", "StitchMAX200v2_aug", "StitchMAX400v2"]:
        (ROOT / name).mkdir(exist_ok=True)
    for entry in p["links"]:
        assert sha(entry["source"]) == entry["sha256"]
        link(entry["source"], ROOT / "StitchMAX200v2" / entry["name"])
    receipt_path = REPORT / "generated.json"
    done = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    by_name = {r["name"]: r for r in done}
    for r in done:
        assert sha(r["output"]) == r["sha256"]
    todo = [t for t in p["tasks"] if t["name"] not in by_name]
    status = {"status": "generating", "completed": len(done), "total": 243, "started": time.time()}
    dump(REPORT / "status.json", status)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context("spawn")) as ex:
            futures = [ex.submit(generate, t) for t in todo]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                done.append(result)
                dump(receipt_path, done)
                status.update(completed=len(done), updated=time.time())
                dump(REPORT / "status.json", status)
                print(json.dumps({"completed": len(done), "name": result["name"], "diagnostics": result["diagnostics"]}), flush=True)
        for pool in ["StitchMAX200v2", "StitchMAX200v2_aug"]:
            files = list((ROOT / pool).glob("*.pkl"))
            assert len(files) == 200, (pool, len(files))
            for f in files:
                assert f.is_file() and f.stat().st_size > 0
                link(f, ROOT / "StitchMAX400v2" / f.name)
        assert len(list((ROOT / "StitchMAX400v2").glob("*.pkl"))) == 400
        summary = {"status": "complete", "completed": time.time(), "counts": p["expected_counts"], "linked_originals": 157, "generated_originals": 43, "generated_augmentations": 200, "hand_shift_m": SHIFT, "max_ik_position_cm": max(r["diagnostics"]["ik_pos_cm"] for r in done), "max_ik_position_post_cm": max(r["diagnostics"]["ik_pos_cm_post"] for r in done), "max_ik_rotation_deg": max(r["diagnostics"]["ik_rot_deg"] for r in done), "validation": "Original body-pose gate, finite tensors, frame counts and grab indices. IK diagnostics recorded; no claim of policy success."}
        for name in ["StitchMAX200v2", "StitchMAX200v2_aug", "StitchMAX400v2"]:
            dump(ROOT / name / "_PROVENANCE.json", {"plan": str(REPORT / "plan.json"), "generated": str(receipt_path), **summary})
        dump(REPORT / "status.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as e:
        status.update(status="failed", error=repr(e), updated=time.time())
        dump(REPORT / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "build"])
    args = parser.parse_args()
    plan() if args.phase == "plan" else build()
