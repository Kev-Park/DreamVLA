"""Launch the authorized nested training-reference subset experiments on bluesclues."""
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path.home() / "kevin"
manifest = json.loads(Path(__file__).with_name("residual_subset_manifest.json").read_text())
usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
free = [int(row.split(",")[0]) for row in usage.splitlines() if int(row.split(",")[1]) < 2000]
needed = []
for name in manifest:
    session = "train_" + name + "_kevin"
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
        needed.append(name)
if len(free) <= len(needed):
    raise SystemExit("Need four free GPUs plus headroom; refusing to fill the cluster")
for name, gpu in zip(needed, free):
    info = manifest[name]
    source = root / "ref_motions" / info["pool"]
    destination = root / "ref_motions" / name
    destination.mkdir(exist_ok=True)
    for filename in info["files"]:
        original = source / filename
        if not original.is_file():
            raise FileNotFoundError(original)
        link = destination / filename
        if not link.exists():
            link.symlink_to(original)
    (destination / "subset_manifest.json").write_text(json.dumps(info, indent=2))
    walk = name.startswith("walk")
    code = root / ("wt/locomotion/DreamVLA" if walk else "DreamVLA")
    activate = "source ~/kevin/wt/locomotion/.venv/bin/activate" if walk else "source ~/miniconda3/etc/profile.d/conda.sh; conda activate dreamcontrol_51"
    expert_flags = " HS_REWORK_CLOSE_ON_ARRIVAL=1 HS_REWORK_PALM_NORMAL=0 HS_REWORK_PALM_REACH=0.07 HS_PALM_NORMAL_SIGN=+1" if walk else ""
    logdir = root / "dline/residual_subsets"
    logdir.mkdir(exist_ok=True)
    command = ("export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; " + activate +
               "; unset HS_REWORK_ARRIVE_FINGER_D; cd " + shlex.quote(str(code / "Training")) +
               "; env HS_REWORK=1 HS_REWORK_OBJ_PC=1 HS_REWORK_OBJ_GATE=1 HS_REWORK_CONTACT_LEAD=2" + expert_flags +
               f" CUDA_VISIBLE_DEVICES={gpu} python scripts/reinforcement_learning/rsl_rl/train_sonic_adapter.py" +
               " --task Isaac-Motion-Tracking-Pick-HOI-v0 --waist-dof 29 --headless --num_envs 1024 --max_iterations 9000" +
               " --run_name " + name + " --seed 42 --sonic-pt ~/kevin/sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000" +
               " --residual-scale 0.1 --residual-transform additive_free --ref-motions-path " + name +
               (" --tracking-scale 2.0" if walk else "") + " > " + shlex.quote(str(logdir / (name + ".log"))) +
               " 2>&1; result=$?; printf '%s\n' \"$result\" > " + shlex.quote(str(logdir / (name + ".exit"))))
    subprocess.run(["tmux", "new-session", "-d", "-s", "train_" + name + "_kevin", command], check=True)
    print(name, "GPU", gpu)
