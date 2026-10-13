"""Supervise direct-token PPO and frozen SONIC reference-grasp evaluation.

The direct actor changes only token composition; PPO, references, observations,
initialization, finger action, and rewards match T20_max_v3_env2048.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import watch_t20max_v3 as common

ROOT, REPO = common.ROOT, common.REPO
OUT = ROOT / "eval_videos/walk_v3_study/sonic"
RUN = "T20_max_v3_direct_env2048"
POOL = "StitchMAX100v3_eval"
MILESTONES = list(range(500, 9000, 500)) + [8999]


def command(script, pool, direct=False):
    cmd = common.command(script, pool)
    if direct:
        cmd[cmd.index("--residual-transform") + 1] = "direct"
    return cmd


def launch(cmd, name, gpu):
    log = OUT / (name + ".log")
    assert not log.exists(), f"Existing log: {log}"
    env = common.environment(gpu)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, cwd=REPO/"Training", env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    common.write(OUT/(name+".launch.json"), {
        "pid": proc.pid, "command": cmd, "gpu": gpu, "started": time.time(),
        "environment": {k:v for k,v in env.items() if k.startswith("HS_")}})
    return proc, log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/"workflow.json").exists(), "Inspect existing workflow before resuming"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    state = {"status": "waiting_for_gpus", "controller_pid": os.getpid(),
             "started": time.time(), "train_gpu": 9, "eval_gpu": 8,
             "run": RUN, "num_envs": 2048, "max_iterations": 9000,
             "eval_pool": POOL, "eval_episodes": 2000, "errors": [],
             "note": "Direct token actor starts from zero tokens, not the frozen encoder token",
             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    train = active = None
    train_dir = None
    completed, rows = set(), []
    last_progress, last_iteration = time.time(), None
    try:
        while True:
            now = time.time()
            state["alerts"] = []
            if train is None and common.gpu_free(9):
                cmd = command("train_sonic_adapter.py", "StitchMAX400v3", True) + [
                    "--num_envs", "2048", "--max_iterations", "9000", "--seed", "42",
                    "--run_name", RUN, "--tracking-scale", "2.0"]
                train, train_log = launch(cmd, "train_direct", 9)
                state.update(status="training", train_pid=train.pid, training_started=now)
            if train:
                text = common.tail(train_log)
                iterations = re.findall(r"Learning iteration\s+(\d+)", text)
                if iterations:
                    iteration = int(iterations[-1])
                    if iteration != last_iteration:
                        last_iteration, last_progress = iteration, now
                    state["latest_iteration"] = iteration
                if "Traceback (most recent call last)" in text:
                    state["alerts"].append("Training traceback; inspect train_direct.log")
                if train.poll() is not None and train.returncode != 0:
                    raise RuntimeError(f"Training exited {train.returncode}: {text[-2000:]}")
                if train.poll() is None and now-last_progress > 900:
                    state["alerts"].append("Training has not advanced for 15 minutes")
                dirs = list((REPO/"Training/logs/rsl_rl/g1_sonic_adapter").glob("*_"+RUN))
                assert len(dirs) <= 1, "Ambiguous training directory"
                if dirs:
                    train_dir = dirs[0]
                    state["checkpoint_dir"] = str(train_dir)
            if active and active[1].poll() is not None:
                key, proc, log, started = active
                if proc.returncode == 0:
                    try:
                        rows.append({"experiment": key, "evaluated_at": now,
                                     **common.metrics(common.tail(log, 2000000))})
                        common.write(OUT/"results.json", rows)
                    except Exception as exc:
                        state["errors"].append({"experiment": key, "error": repr(exc)})
                else:
                    state["errors"].append({"experiment": key, "exit_code": proc.returncode})
                completed.add(key)
                active = None
            ready = []
            if "frozen_reference_grasp" not in completed:
                ready.append(("frozen_reference_grasp", None))
            if train_dir:
                for iteration in MILESTONES:
                    ck = train_dir/f"model_{iteration}.pt"
                    key = f"direct_{iteration}"
                    if key not in completed and ck.exists() and ck.stat().st_size > 0 and now-ck.stat().st_mtime > 10:
                        ready.append((key, ck))
            if active is None and ready and common.gpu_free(8):
                key, ck = ready[0]
                cmd = command("eval_sonic_adapter.py", POOL, ck is not None) + [
                    "--num_envs", "1000", "--num_episodes", "2000", "--seed", "0"]
                cmd += ["--path", str(ck)] if ck else ["--zero-residual", "--reference-grasp"]
                proc, log = launch(cmd, key, 8)
                active = (key, proc, log, now)
            if active and now-active[3] > 1800:
                state["alerts"].append("Evaluation exceeds 30 minutes; inspect log and PID")
            if state["errors"]:
                state["alerts"].append("An evaluation failed; see errors. Training remains supervised.")
            state.update(heartbeat=now, completed_evals=len(rows),
                         active_eval={"name": active[0], "pid": active[1].pid} if active else None)
            common.write(OUT/"workflow.json", state)
            if train and train.poll() is not None and active is None:
                assert train_dir and (train_dir/"model_8999.pt").is_file(), "Final checkpoint missing"
                if len(completed) == 1+len(MILESTONES):
                    if state["errors"]:
                        raise RuntimeError("Training finished with failed evaluations")
                    state.update(status="complete", completed=time.time())
                    common.write(OUT/"workflow.json", state)
                    return
            time.sleep(30)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), heartbeat=time.time())
        common.write(OUT/"workflow.json", state)
        raise


if __name__ == "__main__":
    main()
