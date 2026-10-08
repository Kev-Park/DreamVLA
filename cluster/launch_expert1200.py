"""Launch the authorized 1200-attempt expert data collection pipelines."""
import json
from pathlib import Path
import shlex
import subprocess
root = Path.home() / "kevin"
state = json.loads((root / "dline/watch/state.json").read_text())
for task, oldtag, count, passes, last in (("pick", "af60v8f", 400, 3, 400), ("walk", "T20maxf", 157, 8, 101)):
    tag = "expert" + task + "1200"
    env = dict(state[oldtag]["env"])
    code = Path(env.get("CODE", str(root / "DreamVLA")))
    env.update(TAG=tag, NSH="4", DAGGER_N="0", BASE="keep:" + str(root / "dline" / oldtag / "keep_base.txt"),
               SKIP_CODE_UPDATE="1", EXPERT_BOX=str(root / "dline" / oldtag / "box.json"),
               EXPERT_PASSES=str(passes), EXPERT_REFERENCE_COUNT=str(count), EXPERT_LAST_COUNT=str(last),
               COLLECT_SCRIPT=str(root / "wt/dagger-watch-sparse/Training/scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py"),
               PYTHONPATH=str(code / "Training/scripts/reinforcement_learning/rsl_rl"))
    if task == "walk":
        env["CK"] = str(root / "residuals/T20_max_model_8999.pt")
        env["ENVX"] = "HS_REWORK_CLOSE_ON_ARRIVAL=1 HS_REWORK_PALM_NORMAL=0 HS_REWORK_PALM_REACH=0.07 HS_PALM_NORMAL_SIGN=+1"
    if not Path(env["CK"]).is_file():
        raise FileNotFoundError(env["CK"])
    work = root / "dline" / tag
    work.mkdir(exist_ok=True)
    (work / "environment.json").write_text(json.dumps(env, indent=2))
    session = tag + "_kevin"
    if (work / "done").exists() or subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        continue
    command = "export XLA_PYTHON_CLIENT_PREALLOCATE=false; unset HS_REWORK_ARRIVE_FINGER_D; " + " ".join(
        f"{k}={shlex.quote(v)}" for k, v in env.items()) + " bash ~/kevin/wt/dagger-watch-sparse/cluster/expert1200.sh"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    print("Started", tag, "1200 attempts, four collector slots")
