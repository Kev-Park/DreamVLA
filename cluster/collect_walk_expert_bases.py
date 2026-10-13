"""Collect exact 10/50/200-success walking bases from nested reference pools.

Reuse all retained episodes through symlinks. First sweep the 200 originals with
the deterministic expert; calibrate its palm box once using the existing pick
procedure. Retry failed references with sampled expert actions up to three times,
then fill shortfalls with sampled episodes of successful references in that pool.
No existing files are overwritten, no jobs are killed, and failures are explicit.
"""
import fcntl
import glob
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import watch_t20max_v3 as common

ROOT, REPO = common.ROOT, common.REPO
OUT = ROOT / "eval_videos/walk_v3_study/expert_collection"
DATA = ROOT / "datasets/walk_v3_study"
POOL = ROOT / "ref_motions/StitchMAX200v3"
RSL = REPO / "Training/scripts/reinforcement_learning/rsl_rl"
EXPERT_LINK = ROOT / "eval_videos/T20_max_v3_env2048/early_stop/best_evaluated.pt"
GPUS = [2, 3, 5, 6]


def inspect_rollouts(root, box=None):
    import h5py
    import numpy as np
    sys.path.insert(0, str(RSL))
    from vla_sonic.grasp_success import score
    rows = []
    for file in sorted(root.rglob("*.hdf5")):
        if "_passA" in file.name:
            continue
        match = re.search(r"__motion_(\d+)", file.name)
        assert match, str(file)
        with h5py.File(file, "r", locking=False) as h:
            g = h["data/demo_0"]
            assert len(g["actions"]) > 0
            passed = True
            if box is not None:
                result = score(g["obs/object_pos"][()], g["obs/object_quat"][()],
                               g["obs/robot0_root_pos_w"][()], g["obs/robot0_root_quat_w"][()],
                               g["teleop/right_wrist"][()], box_lo=np.array(box["lo"]),
                               box_hi=np.array(box["hi"]))
                passed = bool(result["success"])
            rows.append({"path": str(file), "motion_id": int(match.group(1)),
                         "frames": len(g["actions"]), "passing": passed})
    return rows


def collect(ids, name, sampled, seed, expert, state):
    directory = DATA / "raw" / name
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, gpu in enumerate(GPUS):
        selected = ids[i::len(GPUS)]
        if not selected:
            continue
        idsfile = directory/f"ids_{i}.txt"
        if idsfile.exists():
            assert list(map(int, idsfile.read_text().split())) == selected, f"Changed shard inputs: {idsfile}"
        else:
            idsfile.write_text(" ".join(map(str, selected))+"\n")
        log = directory/f"shard{i}.log"
        if log.exists():
            text = common.tail(log)
            assert "Collection finished." in text and "errored=0" in text, f"Prior incomplete collector needs reconciliation: {log}"
            continue
        jobs.append({"gpu": gpu, "ids": selected, "idsfile": idsfile, "index": i})
    pending = list(jobs)
    active = []
    while pending or active:
        for job in pending[:]:
            slots = subprocess.check_output([str(common.PYTHON), str(REPO/"cluster/gpu_slots.py"),
                "reserve", "--owner", str(os.getpid()), "--count", "1", "--mb", "17000"], text=True).split()
            if not slots:
                continue
            job["gpu"] = int(slots[0])
            cmd = common.command("collect_sonic_adapter.py", "StitchMAX200v3")
            cmd[cmd.index("--task")+1] = "Isaac-Motion-Tracking-Pick-Cam-HOI-v0"
            cmd += ["--checkpoint-path", str(expert), "--num-samples", str(len(job["ids"])),
                    "--rollout-length", "1200", "--motion-ids", str(job["idsfile"]),
                    "--seed", str(seed), "--output-directory", str(directory/f"shard{job['index']}")]
            if sampled:
                cmd.append("--sample-expert-actions")
            log = directory/f"shard{job['index']}.log"
            env = common.environment(job["gpu"])
            env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            with log.open("w") as stream:
                proc = subprocess.Popen(cmd, cwd=REPO/"Training", env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            job.update(proc=proc, log=log, started=time.time(), command=cmd)
            common.write(directory/f"launch_{job['index']}.json", {
                "pid": proc.pid, "gpu": job["gpu"], "command": cmd,
                "started": job["started"], "sampled": sampled})
            pending.remove(job)
            active.append(job)
        for job in active[:]:
            if job["proc"].poll() is None:
                continue
            text = common.tail(job["log"])
            assert job["proc"].returncode == 0, f"Collection failed: {job['log']}\n{text[-2000:]}"
            assert "Collection finished." in text, f"Missing completion: {job['log']}"
            # The collector can catch per-rollout exceptions: do not accept a clean process
            # exit as proof all requested references were actually attempted.
            assert "errored=0" in text, f"Per-rollout errors: {job['log']}"
            subprocess.run([str(common.PYTHON), str(REPO/"cluster/gpu_slots.py"), "release",
                "--owner", str(os.getpid()), "--gpu", str(job["gpu"])], check=True)
            state.setdefault("completed_jobs", []).append({"name": name, "gpu": job["gpu"],
                "references": job["ids"], "log": str(job["log"]), "completed": time.time()})
            active.remove(job)
        state.update(heartbeat=time.time(), phase=name,
                     active_jobs=[{"pid": j["proc"].pid, "gpu": j["gpu"], "log": str(j["log"]),
                                   "seconds": time.time()-j["started"],
                                   "seconds_since_log_update": time.time()-j["log"].stat().st_mtime}
                                  for j in active],
                     waiting_jobs=len(pending))
        state["alerts"] = ["Collection log has not advanced for 15 minutes"] if any(
            x["seconds_since_log_update"] > 900 for x in state["active_jobs"]) else []
        common.write(OUT/"workflow.json", state)
        if pending or active:
            time.sleep(30)
    return directory


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/"workflow.json").exists():
        old = json.loads((OUT/"workflow.json").read_text())
        assert not Path(f"/proc/{old['controller_pid']}").exists(), "Controller still running"
        assert not old.get("active_jobs"), "Collection resume needs reconciliation"
    else:
        old = None
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    # Match motion_lib_base.load_data's directory glob order exactly. Sorting here
    # would silently assign different reference names to the simulator's IDs.
    refs = [Path(p) for p in glob.glob(str(POOL/"*.pkl"))]
    assert len(refs) == 200
    expert = EXPERT_LINK.resolve(strict=True)
    assert expert.name == "model_4500.pt"
    order = list(range(200))
    random.Random(42).shuffle(order)
    pools = {str(n): sorted(order[:n]) for n in [10, 50, 200]}
    DATA.mkdir(parents=True, exist_ok=True)
    protocol = {
        "expert": str(expert), "reference_pool": str(POOL),
        "reference_files": [str(p) for p in refs], "nested_motion_ids": pools,
        "base_passing_episodes": [10, 50, 200], "dagger_attempts_per_round": [10, 50, 200],
        "dagger_rounds": 3, "global_batch": 16, "effective_epochs": 5.7,
        "initialization": "independent pretrained GR00T on each cumulative dataset",
        "retry": "3 sampled attempts per failed reference, then sampled working references",
        "success_filter": "existing collector physical-hold + fixed calibrated four-rule pick filter",
        "evaluation_pool": str(ROOT/"ref_motions/StitchMAX100v3_eval")}
    if (OUT/"protocol.json").exists():
        assert json.loads((OUT/"protocol.json").read_text()) == protocol, "Collection protocol changed"
    else:
        common.write(OUT/"protocol.json", protocol)
    state = {"status": "collecting", "controller_pid": os.getpid(), "started": time.time(),
             "base_datasets": {}, "errors": [], "alerts": [],
             "expert": str(expert), "gpu_candidates": GPUS}
    if old:
        state.update(old)
        state.update(controller_pid=os.getpid(), status="collecting", resumed=time.time(), alerts=[])
    try:
        # An empty first-attempt directory may remain if the controller only waited
        # for GPUs. Preserve it and use a new name on a reconciled restart.
        if (OUT/"initial_scores.json").exists():
            box = json.loads((OUT/"box.json").read_text())
            rows = json.loads((OUT/"initial_scores.json").read_text())
        else:
            name = "initial_deterministic"
            if (DATA/"raw"/name).exists():
                assert not list((DATA/"raw"/name).glob("shard*.log")), "Prior collector launch needs reconciliation"
                name += "_"+str(time.time_ns())
            initial = collect(order, name, False, 0, expert, state)
            cmd = [str(common.PYTHON), "-m", "vla_sonic.calibrate_box", "--out", str(OUT/"box.json"), str(initial)]
            with (OUT/"calibration.log").open("w") as stream:
                subprocess.run(cmd, cwd=RSL, stdout=stream, stderr=subprocess.STDOUT, check=True)
            box = json.loads((OUT/"box.json").read_text())
            rows = inspect_rollouts(initial, box)
            common.write(OUT/"initial_scores.json", rows)
        passed = [r for r in rows if r["passing"]]
        selected_previous = []
        for n in [10, 50, 200]:
            base = DATA/f"base{n}"
            if (base/"ready").is_file():
                manifest = json.loads((base/"manifest.json").read_text())
                assert manifest["count"] == len(manifest["episodes"]) == n
                assert manifest["selected_reference_ids"] == pools[str(n)] and manifest["box"] == box
                assert all(Path(r["path"]).is_file() for r in manifest["episodes"])
                selected_previous = manifest["episodes"]
                continue
            allowed = set(pools[str(n)])
            selected = list(selected_previous)
            represented = {r["motion_id"] for r in selected}
            selected += [r for r in passed if r["motion_id"] in allowed-represented]
            for attempt in range(1, 4):
                missing = sorted(allowed-{r["motion_id"] for r in selected})[:n-len(selected)]
                if not missing:
                    break
                raw = collect(missing, f"base{n}_retry{attempt}", True, 10000+n*10+attempt, expert, state)
                scored = inspect_rollouts(raw, box)
                common.write(raw/"scores.json", scored)
                selected += [r for r in scored if r["passing"]]
            fill = 0
            while len(selected) < n:
                working = sorted({r["motion_id"] for r in selected})
                assert working, f"No passing references for base{n}; user review needed"
                # Every pass generates at most the deficit and uses distinct references.
                fill += 1
                assert fill <= 30, f"base{n} still short after 30 fill sweeps; inspect failures"
                take = working[:min(n-len(selected), len(working))]
                raw = collect(take, f"base{n}_fill{fill}", True, 100000+n*100+fill, expert, state)
                scored = inspect_rollouts(raw, box)
                common.write(raw/"scores.json", scored)
                selected += [r for r in scored if r["passing"]]
            assert len(selected) == n, (n, len(selected))
            base = DATA/f"base{n}"
            inputs = base/"hdf5"
            inputs.mkdir(parents=True, exist_ok=False)
            for i, row in enumerate(selected):
                source = Path(row["path"])
                (inputs/f"ep_{i:05d}__{source.name}").symlink_to(source)
            common.write(base/"manifest.json", {"episodes": selected, "count": n,
                "distinct_references": len({r["motion_id"] for r in selected}),
                "selected_reference_ids": pools[str(n)], "expert": str(expert), "box": box})
            (base/"ready").write_text("Validated exact passing episode count\n")
            state["base_datasets"][str(n)] = {"path": str(base), "episodes": n,
                "distinct_references": len({r["motion_id"] for r in selected})}
            selected_previous = selected
        state.update(status="complete", completed=time.time(), heartbeat=time.time())
        common.write(OUT/"workflow.json", state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), heartbeat=time.time())
        common.write(OUT/"workflow.json", state)
        raise


if __name__ == "__main__":
    main()
