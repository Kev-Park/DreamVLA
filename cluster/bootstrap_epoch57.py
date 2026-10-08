"""Bootstrap new epoch57 evaluations and round-three continuation from retained data."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

BUDGETS = {"af60v8f": [61000, 97000, 127000], "T20maxf": [26000, 41000, 59000]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    root = Path.home() / "kevin"
    oldtag = next(t for t in BUDGETS if args.run.startswith(t + "57_"))
    tag = oldtag + "57"
    old = root / "dline" / oldtag
    work = root / "dline" / tag
    work.mkdir(exist_ok=True)
    (work / "ftq").mkdir(exist_ok=True)
    state = json.loads((root / "dline/watch/state.json").read_text())
    env = dict(state[oldtag]["env"])
    for filename in ("box.json", "keep_base.txt", "keep_d1.txt", "keep_d2.txt"):
        target = work / filename
        if not target.exists():
            shutil.copy2(old / filename, target)
    for r, steps in enumerate(BUDGETS[oldtag]):
        run = tag + ("_run01" if r == 0 else f"_dagger{r}_run01")
        (work / ("steps_" + run)).write_text(str(steps) + "\n")
    for marker in ("base.done", "r0.done", "r1.done", "r2.done"):
        (work / marker).touch()
    if not (work / "finetune_policy.json").exists():
        (work / "finetune_policy.json").write_text(json.dumps({"passes": 5.7, "max_steps": 0}))
    code = Path(os.path.expanduser(env.get("CODE", "~/kevin/DreamVLA")))
    scriptdir = root / "wt/dagger-watch-sparse/Training/scripts/reinforcement_learning/rsl_rl"
    env.update(TAG=tag, BASE="keep:" + str(old / "keep_base.txt"), BOX=str(old / "box.json"),
               NR="5", FT_EXTERNAL="1", FT_PASSES="5.7", FT_MAX_STEPS="0", DAGGER_N="200",
               DAGGER_SAMPLE_ALL="1" if oldtag == "af60v8f" else "0", START_ROUND="3", MONTAGE="0",
               SKIP_CODE_UPDATE="1", EVAL_SCRIPT=str(scriptdir / "eval_vla_sonic.py"),
               COLLECT_SCRIPT=str(scriptdir / "collect_sonic_adapter.py"),
               PYTHONPATH=str(code / "Training/scripts/reinforcement_learning/rsl_rl"))
    if oldtag == "T20maxf":
        env["CK"] = str(root / "residuals/T20_max_model_8999.pt")
        env["ENVX"] = "HS_REWORK_CLOSE_ON_ARRIVAL=1 HS_REWORK_PALM_NORMAL=0 HS_REWORK_PALM_REACH=0.07 HS_PALM_NORMAL_SIGN=+1"
    (work / "epoch57_environment.json").write_text(json.dumps(env, indent=2))
    prefix = "export XLA_PYTHON_CLIENT_PREALLOCATE=false; unset HS_REWORK_ARRIVE_FINGER_D; "
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    script = str(root / "wt/dagger-watch-sparse/cluster/dagger_line.sh")
    def launch(session, extra):
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            return
        subprocess.run(["tmux", "new-session", "-d", "-s", session,
                        prefix + assignments + extra + " bash " + shlex.quote(script)], check=True)
    if not (work / ("ev_" + args.run + ".done")).exists():
        launch("ev_" + args.run + "_kevin", " EVAL_ONLY=" + shlex.quote(args.run))
    if "_dagger2_" in args.run:
        if not (work / "done").exists():
            launch("dline_v857_kevin" if oldtag == "af60v8f" else "dline_t2057_kevin", "")
    print("Bootstrapped", args.run)


if __name__ == "__main__":
    main()
