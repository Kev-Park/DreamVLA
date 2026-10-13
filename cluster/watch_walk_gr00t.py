"""Read-only health and completion checks for all three walking GR00T chains."""
import fcntl
import json
import os
from pathlib import Path
import time

from gr00t_budget_chain import atomic, checkpoint_ok, read

ROOT = Path.home()/"kevin"
OUT = ROOT/"eval_videos/walk_v3_study/gr00t"


def process(pid):
    p = Path("/proc")/str(pid)/"cmdline"
    try:
        return p.read_text().replace("\0", " ")
    except OSError:
        return None


def inspect(budget):
    tag = f"walk{budget}_v3_dagger57_20261013"
    work = ROOT/"dline"/tag
    row = {"budget": budget, "tag": tag, "alerts": [], "rounds": []}
    launch = work/"controller.launch.json"
    if not launch.exists():
        row["status"] = "not_launched"
        return row
    handle = read(launch)
    cmd = process(handle["pid"])
    row.update(controller_pid=handle["pid"], controller_command=cmd)
    if (work/"abort.json").exists():
        row.update(status="failed", failure=read(work/"abort.json"))
        return row
    if cmd is None and not (work/"done.json").exists():
        row["alerts"].append("Controller exited")
    for stage in work.glob("*.process.json"):
        job = read(stage)
        log = stage.with_name(stage.name.replace(".process.json", ".log"))
        live = process(job["pid"])
        if live:
            row.setdefault("active_workers", []).append(dict(job, command_live=live,
                seconds_since_log_update=time.time()-log.stat().st_mtime if log.exists() else None))
            if log.exists() and time.time()-log.stat().st_mtime > 1800:
                row["alerts"].append("Stage log stalled: "+log.name)
    for index in range(4):
        run = f"{tag}_r{index}"
        trained, evaluated = work/(run+".trained.json"), work/(run+".eval.json")
        item = {"round": index, "run": run, "trained": False, "evaluated": False}
        if trained.exists():
            item.update(training=read(trained), trained=True)
            if not checkpoint_ok(Path(item["training"]["checkpoint"])):
                row["alerts"].append("Checkpoint shards missing: "+run)
        if evaluated.exists():
            item.update(evaluation=read(evaluated), evaluated=True)
            assert item["evaluation"]["episodes"] == 100
            assert sorted(r["motion_id"] for r in item["evaluation"]["per_episode"]) == list(range(100))
        row["rounds"].append(item)
    row["status"] = "running" if cmd else "stopped_without_completion"
    if (work/"done.json").exists():
        assert all(r["trained"] and r["evaluated"] for r in row["rounds"])
        assert not row["alerts"], row["alerts"]
        row["status"] = "complete"
    elif cmd and (work/"waiting_for_base.json").exists() and not (work/"recipe.json").exists():
        base = ROOT/f"datasets/walk_v3_study/base{budget}"
        if not (base/"ready").exists():
            row["status"] = "waiting_for_exact_expert_base"
        elif any("convert_sharded.sh" in p["command_live"] for p in row.get("active_workers", [])):
            row["status"] = "converting_base"
        else:
            row["status"] = "preparing_base_training"
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"health.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            rows = [inspect(n) for n in [10, 50, 200]]
            state = {"heartbeat": time.time(), "watcher_pid": os.getpid(), "chains": rows,
                     "status": "complete" if all(r["status"] == "complete" for r in rows) else "monitoring"}
            atomic(OUT/"workflow.json", state)
            if state["status"] == "complete":
                return
        except BaseException as exc:
            atomic(OUT/"health_failure.json", {"time": time.time(), "error": repr(exc), "pid": os.getpid()})
            raise
        time.sleep(30)


if __name__ == "__main__":
    main()
