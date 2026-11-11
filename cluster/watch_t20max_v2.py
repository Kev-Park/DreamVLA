"""Supervise the authorized 400-reference / 2048-environment T20_max experiment.

Run in the locomotion dreamcontrol_51 overlay on Bluesclues, inside tmux.
Never kills jobs, overwrites old runs, deletes checkpoints, or retries failed jobs.
"""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path.home() / "kevin"
REPO = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_videos/T20_max_v2_env2048"
BUILD = ROOT / "eval_videos/StitchMAX200v2/status.json"
PYTHON = ROOT / "wt/locomotion/.venv/bin/python"
RUN = "T20_max_v2_env2048"
ENVS, HORIZON, ITERATIONS = 2048, 24, 9000
MILESTONES = list(range(500, 9000, 500)) + [8999]
POOLS = ["StitchMAX200", "StitchMAX400v2", "StitchMAX100v2_eval"]


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def tail(path, size=200000):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - size))
        return f.read().decode(errors="replace")


def metrics(text):
    # Match the authoritative physical 5cm hold lines, not the reward counter.
    held = re.search(r"\[PHYS\] held \(lift>=5cm[^\n]*?:\s*(\d+) / (\d+) = ([\d.]+)%", text)
    upright = re.search(r"\[PHYS\] held-upright[^\n]*lift>=5cm\s+(\d+) / (\d+) = ([\d.]+)%", text)
    assert held and upright and "[eval_sonic_adapter] DONE" in text, "Incomplete evaluation metrics"
    successes, episodes = map(int, held.group(1, 2))
    assert episodes >= 2000, episodes
    p, z = successes / episodes, 1.96
    center = (p + z*z/(2*episodes)) / (1 + z*z/episodes)
    radius = z*math.sqrt(p*(1-p)/episodes + z*z/(4*episodes*episodes)) / (1+z*z/episodes)
    return {"episodes": episodes, "held5_successes": successes, "held5_pct": 100*p,
            "held5_ci95_low": 100*(center-radius), "held5_ci95_high": 100*(center+radius),
            "held_upright5_pct": 100*int(upright.group(1))/int(upright.group(2))}


def environment(gpu):
    # Remove inherited experimental overrides, then use the original palm-only recipe.
    env = {k: v for k, v in os.environ.items() if not k.startswith("HS_")}
    env.update(HS_REWORK="1", HS_REWORK_OBJ_PC="1", HS_REWORK_OBJ_GATE="1",
               HS_REWORK_CONTACT_LEAD="2", HS_REWORK_CLOSE_ON_ARRIVAL="1",
               HS_REWORK_ARRIVE_FINGER_D="0", HS_REWORK_ARRIVE_D="0.08",
               HS_REWORK_PALM_NORMAL="0", HS_REWORK_PALM_REACH="0.07",
               HS_PALM_NORMAL_SIGN="+1", HYBRID_RARM="0",
               CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    return env


def gpu_free(index):
    row = subprocess.check_output(["nvidia-smi", f"--id={index}",
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    memory, util = [int(x.strip()) for x in row.strip().split(",")]
    return memory < 3000 and util < 10


def command(script, pool):
    return [str(PYTHON), "-u", "scripts/reinforcement_learning/rsl_rl/"+script,
        "--task", "Isaac-Motion-Tracking-Pick-HOI-v0", "--headless", "--waist-dof", "29",
        "--sonic-pt", str(ROOT/"sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000"),
        "--residual-scale", "0.1", "--residual-transform", "additive_free",
        "--ref-motions-path", str(ROOT/"ref_motions"/pool)]


def launch(cmd, name, gpu):
    log = OUT / (name+".log")
    assert not log.exists(), f"Refusing to overwrite {log}"
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO/"Training", env=environment(gpu),
                                stdout=f, stderr=subprocess.STDOUT)
    write(OUT/(name+".launch.json"), {"pid": proc.pid, "command": cmd,
        "environment": {k:v for k,v in environment(gpu).items() if k.startswith("HS_") or k in
                        ["HYBRID_RARM", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
        "started": time.time(), "gpu": gpu})
    return proc, log


def report(rows):
    write(OUT/"results.json", rows)
    if not rows:
        return
    with (OUT/"learning_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    baselines = {r["pool"]: r for r in rows if r["model"] == "original_T20_max_1024"}
    comparisons = {}
    for pool, baseline in baselines.items():
        candidates = sorted((r for r in rows if r["model"] == RUN and r["pool"] == pool),
                            key=lambda r: r["checkpoint_iteration"])
        reached = [r for r in candidates if r["held5_pct"] >= baseline["held5_pct"]]
        comparisons[pool] = {"baseline": baseline,
            "first_observed_at_or_above_baseline": reached[0] if reached else None,
            "note": "Observed point estimates with 95% Wilson intervals; one seed, not statistical equivalence. Only StitchMAX100v2_eval has held-out pick IDs; read its provenance for walking-prefix overlap."}
    write(OUT/"comparisons.json", comparisons)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/"workflow.json").exists(), "Existing workflow: inspect before resuming"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    state = {"status": "waiting_for_dataset", "controller_pid": os.getpid(),
        "num_envs": ENVS, "horizon": HORIZON, "transitions_per_update": ENVS*HORIZON,
        "minibatches": 4, "transitions_per_minibatch": ENVS*HORIZON//4,
        "max_iterations": ITERATIONS, "eval_every_checkpoint_iterations": 500,
        "eval_episodes": 2000, "eval_seed": 0, "train_gpu": 9, "eval_gpu": 7,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "started": time.time(), "errors": [], "alerts": []}
    train = active_eval = None
    train_log = OUT/"train.log"
    train_dir = None
    train_finished = False
    rows, queued, processed = [], [], set()
    baseline = ROOT/"residuals/T20_max_model_8999.pt"
    assert baseline.is_file() and baseline.stat().st_size > 0
    last_progress, previous_progress = time.time(), None
    try:
        while True:
            now = time.time()
            data = json.loads(BUILD.read_text()) if BUILD.exists() else {}
            if data.get("status") == "failed":
                raise RuntimeError("Dataset build failed: "+str(data))
            ready = data.get("status") == "complete"
            state["dataset"] = data
            if ready and train is None:
                assert all(len(list((ROOT/"ref_motions"/p).glob("*.pkl"))) == n
                    for p,n in [("StitchMAX200v2",200),("StitchMAX200v2_aug",200),("StitchMAX400v2",400)])
                if gpu_free(9):
                    cmd = command("train_sonic_adapter.py", "StitchMAX400v2") + [
                        "--num_envs", str(ENVS), "--max_iterations", str(ITERATIONS),
                        "--seed", "42", "--run_name", RUN, "--tracking-scale", "2.0"]
                    train, train_log = launch(cmd, "train", 9)
                    state.update(status="training", train_pid=train.pid, training_started=now)
            if train is not None:
                text = tail(train_log)
                iterations = re.findall(r"Learning iteration\s+(\d+)", text)
                if iterations:
                    state["latest_iteration"] = int(iterations[-1])
                    if state["latest_iteration"] != previous_progress:
                        previous_progress = state["latest_iteration"]
                        last_progress = now
                    state["seconds_since_iteration_progress"] = now-last_progress
                    if now-last_progress > 900:
                        state["alerts"] = ["Training has not advanced for 15 minutes; inspect train.log and PID. No automatic kill."]
                elif now-state["training_started"] > 900:
                    state["alerts"] = ["Training has not produced its first iteration in 15 minutes."]
                dirs = list((REPO/"Training/logs/rsl_rl/g1_sonic_adapter").glob("*_"+RUN))
                assert len(dirs) <= 1, "Ambiguous run directory"
                if dirs:
                    train_dir = dirs[0]
                    state["checkpoint_dir"] = str(train_dir)
                if train.poll() is not None:
                    if train.returncode != 0:
                        raise RuntimeError(f"Training exited {train.returncode}: {text[-3000:]}")
                    assert train_dir and (train_dir/"model_8999.pt").is_file(), "Final checkpoint missing"
                    train_finished = True
                    state["status"] = "finishing_evaluations"
            for pool in POOLS:
                if pool == "StitchMAX400v2" and not ready:
                    continue
                if pool == "StitchMAX100v2_eval":
                    manifest = ROOT/"ref_motions"/pool/"_PROVENANCE.json"
                    if not manifest.exists() or json.loads(manifest.read_text()).get("status") != "complete":
                        continue
                key = ("original_T20_max_1024", 8999, pool)
                if key not in processed and key not in [x[0] for x in queued] and (not active_eval or active_eval[0] != key):
                    queued.append((key, baseline))
                if train_dir:
                    for it in MILESTONES:
                        ck = train_dir/f"model_{it}.pt"
                        key = (RUN, it, pool)
                        if ck.is_file() and ck.stat().st_size > 0 and now-ck.stat().st_mtime > 10 and key not in processed and key not in [x[0] for x in queued] and (not active_eval or active_eval[0] != key):
                            queued.append((key, ck))
            if active_eval and active_eval[2].poll() is not None:
                key, checkpoint, proc, logfile, started = active_eval
                try:
                    assert proc.returncode == 0, f"Eval exit {proc.returncode}; {logfile}"
                    result = metrics(tail(logfile, 2000000))
                    n = 1024 if key[0] == "original_T20_max_1024" else ENVS
                    row = {"model": key[0], "checkpoint_iteration": key[1], "ppo_updates": key[1]+1,
                        "training_transitions": (key[1]+1)*n*HORIZON, "pool": key[2],
                        "checkpoint": str(checkpoint), "evaluated_at": now,
                        "training_wall_seconds_to_checkpoint": None if key[0] == "original_T20_max_1024" else checkpoint.stat().st_mtime-state["training_started"],
                        **result}
                    rows.append(row)
                    report(rows)
                except Exception as e:
                    state["errors"].append({"evaluation": key, "error": str(e), "time": now})
                processed.add(key)
                active_eval = None
            if active_eval is None and queued and gpu_free(7):
                key, checkpoint = queued.pop(0)
                cmd = command("eval_sonic_adapter.py", key[2]) + ["--num_envs", "1000",
                    "--num_episodes", "2000", "--seed", "0", "--path", str(checkpoint)]
                proc, logfile = launch(cmd, f"eval_{key[0]}_{key[1]}_{key[2]}", 7)
                active_eval = (key, checkpoint, proc, logfile, now)
            state.update(heartbeat=now, completed_evals=len(rows), pending_evals=len(queued),
                active_eval={"key": active_eval[0], "pid": active_eval[2].pid,
                             "seconds": now-active_eval[4]} if active_eval else None)
            if active_eval and now-active_eval[4] > 1800:
                state["alerts"].append("Evaluation running over 30 minutes; inspect its log and PID.")
            if not ready and data.get("updated", data.get("started")) and now-data.get("updated", data.get("started")) > 900:
                state["alerts"].append("Dataset generation has not advanced in 15 minutes.")
            write(OUT/"workflow.json", state)
            if train_finished and not queued and active_eval is None:
                expected = len(POOLS)*(1+len(MILESTONES))
                if len(rows) == expected:
                    state.update(status="complete", completed=time.time())
                    write(OUT/"workflow.json", state)
                    return
                if state["errors"]:
                    raise RuntimeError(f"Evaluation errors: expected {expected} successful evals, got {len(rows)}")
                state["status"] = "waiting_for_heldout_eval_pool"
            time.sleep(30)
    except BaseException as e:
        state.update(status="failed", error=repr(e), heartbeat=time.time())
        write(OUT/"workflow.json", state)
        raise


if __name__ == "__main__":
    main()
