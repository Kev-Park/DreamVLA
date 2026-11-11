"""Replace the old collection controller only between completed GPU workers."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import watch_t20max_v3 as common

ROOT, REPO = common.ROOT, common.REPO
OUT = ROOT/"eval_videos/walk_v3_study/expert_collection"


def command(pid):
    try:
        return (Path("/proc")/str(pid)/"cmdline").read_text().replace("\0", " ")
    except OSError:
        return ""


def worker_live(pid):
    return "collect_sonic_adapter.py" in command(pid)


def main():
    old = json.loads((OUT/"workflow.json").read_text())
    pid = old["controller_pid"]
    assert str(REPO/"cluster/collect_walk_expert_bases.py") in command(pid)
    deadline = time.time()+1800
    while time.time() < deadline:
        current = json.loads((OUT/"workflow.json").read_text())
        assert current["controller_pid"] == pid, "Controller changed independently"
        if any(worker_live(j["pid"]) for j in current.get("active_jobs", [])):
            time.sleep(.2)
            continue
        assert command(pid), "Controller exited; inspect failure before restarting"
        os.kill(pid, signal.SIGSTOP)
        paused = json.loads((OUT/"workflow.json").read_text())
        try:
            children = (Path("/proc")/str(pid)/"task"/str(pid)/"children").read_text().split()
            if any(worker_live(c) for c in children) or any(worker_live(j["pid"]) for j in paused.get("active_jobs", [])):
                os.kill(pid, signal.SIGCONT)
                time.sleep(.2)
                continue
            for job in paused.get("active_jobs", []):
                text = common.tail(Path(job["log"]))
                assert "Collection finished." in text and "errored=0" in text, job
            receipt = {"old_pid": pid, "time": time.time(), "paused_state": paused,
                       "children": [command(c) for c in children], "gpu_workers_live": False}
            common.write(OUT/("controller_handoff_"+str(time.time_ns())+".json"), receipt)
            paused["active_jobs"] = []
            common.write(OUT/"workflow.json", paused)
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        except BaseException:
            if command(pid):
                os.kill(pid, signal.SIGCONT)
            raise
        for _ in range(100):
            if not command(pid):
                break
            time.sleep(.1)
        assert not command(pid)
        for _ in range(100):
            if subprocess.run(["tmux", "has-session", "-t", "walk_expert_collect_kevin"], capture_output=True).returncode:
                break
            time.sleep(.1)
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        cmd = "exec "+shlex.quote(str(common.PYTHON))+" -u "+shlex.quote(str(REPO/"cluster/collect_walk_expert_bases.py"))+" >> "+shlex.quote(str(OUT/"controller.log"))+" 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", "walk_expert_collect_kevin", cmd], env=env, check=True)
        common.write(OUT/"handoff_complete.json", {"old_pid": pid, "completed": time.time(), "gpu_workers_stopped": 0})
        return
    common.write(OUT/"handoff_wait.json", {"old_pid": pid, "time": time.time(), "reason": "No safe worker boundary observed; original controller untouched"})


if __name__ == "__main__":
    main()
