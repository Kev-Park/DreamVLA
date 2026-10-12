"""Reconstruct walk/pick pairs from the SAME source ID, with train/eval separation.

No training is launched. Existing references and datasets are never overwritten.
Walking intermediates use the original Adapter B lower-body recipe; its unused arm
refinement is skipped because stitch_static_approach replaces that arm completely.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
import build_stitchmax_v2 as shared
import stitch_static_approach as stitcher

ROOT = shared.ROOT
REPORT = ROOT.parent / "eval_videos/StitchOwn_v3"
SOURCES = ROOT / "StitchOwn_v3_sources"
WALKS = ROOT / "WalkOwn300v3"
POOLS = {"train": "StitchMAX200v3", "aug": "StitchMAX200v3_aug", "eval": "StitchMAX100v3_eval"}
COUNTS = {"train": 200, "aug": 200, "eval": 100}
ALIGNMENT = "rigid_xy_final_footfall_to_own_static_grab_v1"
ENV = {"HS_PKL_DOF": "29", "HS_STAND_LOWER": "0", "HS_NO_LEADIN": "1",
       "HS_REFINE_ARM": "0", "HS_RETABLE_Z": "0.911", "HS_FREEZE_FOR": "10",
       "HS_STAND_MIN_STANDOFF": "0.35", "HS_STAND_LAT_BAND": "-0.28,-0.15",
       "HS_REFINE_AFTER_LEADIN": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
       "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1"}


def status(**data):
    shared.dump(REPORT/"status.json", {"updated": time.time(), **data})


def convert(row):
    raw = SOURCES/"raw"/f"pick_{row['id']}.npz"
    assert shared.sha(raw) == row["sha256"]
    output = WALKS/f"pick_{row['id']}.pkl"
    receipt = REPORT/"walk_receipts"/f"{row['id']}.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        assert shared.sha(output) == result["output_sha256"]
        return result
    assert not output.exists(), f"Unreceipted output: {output}"
    env = {k:v for k,v in os.environ.items() if not k.startswith("HS_")}
    env.update(ENV)
    log = REPORT/"walk_logs"/f"{row['id']}.log"
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("holosoma_to_pkl.py")),
           str(raw), str(output.with_suffix(""))]
    with log.open("w") as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert p.returncode == 0 and output.exists(), f"Walk conversion failed: {log}"
    result = {**row, "output": str(output), "output_sha256": shared.sha(output),
              "conversion_environment": ENV}
    shared.dump(receipt, result)
    return result


def walks():
    rows = json.loads((SOURCES/"manifest.json").read_text())["rows"]
    assert len(rows) == 300 and len({r["id"] for r in rows}) == 300
    for p in [WALKS, REPORT/"walk_logs", REPORT/"walk_receipts"]:
        p.mkdir(parents=True, exist_ok=True)
    status(phase="walk_conversion", completed=0, total=300)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(convert, r) for r in rows]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status(phase="walk_conversion", completed=n, total=300)
            print(json.dumps({"walk": result["id"], "completed": n}), flush=True)
    assert len(list(WALKS.glob("*.pkl"))) == 300
    status(phase="walks_complete", completed=300, total=300)


def shifted_walk(walk, shift, root_shift=(0, 0, 0)):
    w = dict(walk, O=walk["O"].copy())
    w["O"][:, :3] += np.asarray(shift)
    root_shift = np.asarray(root_shift)
    assert root_shift.shape == (3,) and root_shift[2] == 0
    for key in ["P", "la", "ra", "hand", "torso"]:
        w[key] = walk[key] + root_shift
    assert np.allclose(np.diff(w["P"], axis=0), np.diff(walk["P"], axis=0), atol=1e-12)
    assert stitcher.final_footfall(w) == stitcher.final_footfall(walk)
    return w


def alignment_shift(w, s):
    fw = stitcher.final_footfall(w)
    dw, _ = stitcher.body_offset(w, fw, w["g"])
    ds, _ = stitcher.body_offset(s, s["g"], s["g"])
    yaw = stitcher.yaw_of(w["Q"][fw])
    forward, left = ds - dw
    return np.array([forward*np.cos(yaw)-left*np.sin(yaw),
                     forward*np.sin(yaw)+left*np.cos(yaw), 0.0])


def align():
    plan(aligned=True)


def plan(aligned=False):
    c, _ = shared.chain()
    rows = json.loads((SOURCES/"manifest.json").read_text())["rows"]
    tasks, failures = [], []
    for row in rows:
        ident = row["id"]
        wp = WALKS/f"pick_{ident}.pkl"
        w = stitcher.load(str(wp), c)
        static_pool = ROOT/("Holosoma_Pick_29_latband60max" if row["split"] == "train" else "Holosoma_Pick_29_latband60max_eval")
        base_path = static_pool/f"pick_{ident}.pkl"
        base = stitcher.load(str(base_path), c)
        root_shift = alignment_shift(w, base) if aligned else np.zeros(3)
        refs = [(row["split"], base_path)]
        if row["split"] == "train":
            a = list(static_pool.glob(f"pick_{ident}_*.pkl"))
            assert len(a) == 1, (ident, a)
            refs.append(("aug", a[0]))
        for split, sp in refs:
            s = stitcher.load(str(sp), c)
            # Carry the augmentation's object offset into this SAME walking scene.
            # Merely anchoring both grasps onto the unshifted bottle would cancel it.
            delta = s["O"][s["g"], :3] - base["O"][base["g"], :3]
            angle = stitcher.yaw_of(w["Q"][stitcher.final_footfall(w)]) - stitcher.yaw_of(base["Q"][base["g"]])
            cs, sn = np.cos(angle), np.sin(angle)
            rotation = np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
            delta = rotation @ delta
            ww = shifted_walk(w, delta, root_shift)
            pos, yaw, ok = shared.gate(ww, s)
            t = {"id": ident, "split": split, "name": sp.name, "walk": str(wp),
                 "walk_sha256": shared.sha(wp), "raw_sha256": row["sha256"],
                 "static": str(sp), "static_sha256": shared.sha(sp),
                 "alignment": ALIGNMENT if aligned else None,
                 "root_alignment_shift_m": root_shift.tolist(),
                 "object_shift_m": delta.tolist(), "body_position_error_m": pos,
                 "body_yaw_error_deg": float(np.degrees(yaw)),
                 "output": str(ROOT/POOLS[split]/sp.name)}
            (tasks if ok else failures).append(t)
    result = {"created": time.time(), "tasks": tasks, "failures": failures,
              "counts": {k:sum(t["split"]==k for t in tasks) for k in POOLS},
              "same_source_only": True, "hand_shift_m": shared.SHIFT,
              "walking_conversion": ENV, "expected": COUNTS,
              "alignment": ALIGNMENT if aligned else None}
    shared.dump(REPORT/("aligned_plan.json" if aligned else "plan.json"), result)
    status(phase="planned" if not failures else "blocked_pose_gate", **{k:result[k] for k in ["counts","failures"]})
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'}), flush=True)


def placement():
    """Read-only proposal: quantify a rigid alignment without changing any reference."""
    c, _ = shared.chain()
    p = json.loads((REPORT/"plan.json").read_text())
    rows = []
    for t in p["tasks"] + p["failures"]:
        if t["split"] == "aug":
            continue
        w, s = stitcher.load(t["walk"], c), stitcher.load(t["static"], c)
        fw = stitcher.final_footfall(w)
        dw, _ = stitcher.body_offset(w, fw, w["g"])
        ds, _ = stitcher.body_offset(s, s["g"], s["g"])
        yaw = stitcher.yaw_of(w["Q"][fw])
        forward, left = ds-dw
        dx = forward*np.cos(yaw)-left*np.sin(yaw)
        dy = forward*np.sin(yaw)+left*np.cos(yaw)
        trial = dict(w, P=w["P"]+np.array([dx,dy,0]))
        pos, angle, ok = shared.gate(trial,s)
        rows.append({"id":t["id"],"split":t["split"],"forward_cm":100*float(forward),
                     "left_cm":100*float(left),"world_dx_cm":100*float(dx),"world_dy_cm":100*float(dy),
                     "distance_cm":100*float(np.hypot(dx,dy)),"footfall_frame":fw,
                     "grab_frame":w["g"],"old_position_error_cm":100*t["body_position_error_m"],
                     "old_yaw_error_deg":t["body_yaw_error_deg"],
                     "proposed_position_error_cm":100*pos,"proposed_yaw_error_deg":float(np.degrees(angle)),
                     "proposed_gate_pass":bool(ok)})
    rows.sort(key=lambda r:(r["split"],r["id"]))
    with (REPORT/"placement_proposal.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    summary={"status":"proposal_only_not_applied","count":len(rows),
             "translation_cm_quantiles":np.quantile([r["distance_cm"] for r in rows],[0,.5,.9,1]).tolist(),
             "proposed_gate_pass":sum(r["proposed_gate_pass"] for r in rows),
             "worst":sorted(rows,key=lambda r:r["distance_cm"],reverse=True)[:10],"rows":rows}
    shared.dump(REPORT/"placement_proposal.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


def generate(t):
    c, lim = shared.chain()
    assert shared.sha(t["walk"]) == t["walk_sha256"]
    assert shared.sha(t["static"]) == t["static_sha256"]
    assert t["alignment"] == ALIGNMENT
    original = stitcher.load(t["walk"], c)
    w = shifted_walk(original, t["object_shift_m"], t["root_alignment_shift_m"])
    s = stitcher.load(t["static"], c)
    stitcher.HAND_SHIFT = np.asarray(shared.SHIFT)
    stitcher.SHIFT_RAMP_FRAMES = None
    pkl, diag = stitcher.stitch(w, s, c, lim)
    assert pkl is not None
    j, pos = np.asarray(pkl["joints"]), np.asarray(pkl["global_position"])
    assert j.shape[1] == 29 and 0 < pkl["grab_idx"] < len(j)
    assert all(np.isfinite(np.asarray(pkl[k])).all() for k in ["joints","global_position","object_poses"])
    fw = diag["footfall"]
    assert np.allclose(j[:fw,:12], w["J"][:fw,:12], atol=1e-6)
    assert np.allclose(pos[:fw], w["P"][:fw], atol=1e-6)
    q = np.asarray(pkl["global_pose"].rotation().wxyz)
    assert np.allclose(q[:fw], original["Q"][:fw], atol=1e-6)
    assert np.isfinite(q).all() and np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5)
    assert all(np.isfinite(v) for v in diag.values())
    p = Path(t["output"])
    assert not p.exists()
    tmp = p.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(pkl, f)
    tmp.replace(p)
    return {**t, "sha256": shared.sha(p), "frames": len(j), "diagnostics": diag,
            "own_walk_root_and_legs_preserved_before_footfall": True}


def build():
    p = json.loads((REPORT/"aligned_plan.json").read_text())
    assert p["alignment"] == ALIGNMENT
    assert not p["failures"], "Pose gate failures require review; no cross-motion fallback"
    assert p["counts"] == COUNTS
    for name in list(POOLS.values()) + ["StitchMAX400v3"]:
        (ROOT/name).mkdir(exist_ok=True)
    receipt = REPORT/"generated.json"
    done = json.loads(receipt.read_text()) if receipt.exists() else []
    for t in done:
        assert shared.sha(t["output"]) == t["sha256"]
    completed = {t["output"] for t in done}
    planned = {t["output"]: t for t in p["tasks"]}
    for t in done:
        assert all(t[k] == v for k, v in planned[t["output"]].items())
    todo = [t for t in p["tasks"] if t["output"] not in completed]
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = [ex.submit(generate,t) for t in todo]
        for future in as_completed(futures):
            t = future.result()
            done.append(t)
            shared.dump(receipt, done)
            status(phase="stitching", completed=len(done), total=500)
            print(json.dumps({"completed": len(done), "id": t["id"], "split": t["split"], "diagnostics": t["diagnostics"]}), flush=True)
    for split, name in POOLS.items():
        assert len(list((ROOT/name).glob("*.pkl"))) == COUNTS[split]
        if split != "eval":
            for f in (ROOT/name).glob("*.pkl"):
                shared.link(f, ROOT/"StitchMAX400v3"/f.name)
    summary = {"phase": "assembled_pending_replay_validation", "counts": COUNTS,
               "training": 400, "same_source_only": True, "hand_shift_m": shared.SHIFT,
               "alignment": ALIGNMENT,
               "plan": str(REPORT/"aligned_plan.json"), "receipt": str(receipt),
               "training_authorized_to_start": False}
    for name in list(POOLS.values()) + ["StitchMAX400v3"]:
        shared.dump(ROOT/name/"_PROVENANCE.json", summary)
    status(**summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["walks","plan","align","build","placement"])
    args = ap.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    try:
        globals()[args.phase]()
    except BaseException as e:
        status(phase="failed", operation=args.phase, error=repr(e))
        raise
