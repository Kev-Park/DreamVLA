"""Reuse passing expert demos for nested 10/50-episode GR00T base datasets."""
import json
import random
from pathlib import Path
import shlex
import subprocess
root = Path.home() / "kevin"
for task, oldtag in (("pick", "af60v8f"), ("walk", "T20maxf")):
    source = [Path(p) for p in (root / "dline" / oldtag / "keep_base.txt").read_text().splitlines() if p]
    chosen = random.Random(42).sample(source, 50)
    for count in (10, 50):
        name = task + str(count)
        work = root / "dline" / name
        work.mkdir(exist_ok=True)
        (work / "ftq").mkdir(exist_ok=True)
        data = root / "datasets" / (name + "_base5p7")
        inputs = data / "hdf5"
        inputs.mkdir(parents=True, exist_ok=True)
        for i, filename in enumerate(chosen[:count]):
            if not filename.is_file():
                raise FileNotFoundError(filename)
            link = inputs / (f"{i:05d}__" + filename.name)
            if not link.exists():
                link.symlink_to(filename)
        (work / "subset_manifest.json").write_text(json.dumps({"seed": 42, "source": oldtag,
            "episodes": [str(p) for p in chosen[:count]], "effective_epochs": 5.7}, indent=2))
        session = "convert_" + name + "_kevin"
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            continue
        script = root / "wt/dagger-watch-sparse/cluster/convert_sharded.sh"
        converter = ("export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; " +
            "bash " + shlex.quote(str(script)) + " --hdf5-root " + shlex.quote(str(inputs)) +
            " --output-path " + shlex.quote(str(data / "lerobot")) +
            ' --dataset-name ds --task-prompt "pick up the mustard bottle" --fps 50 --shards ' + str(min(count, 8)) +
            " > " + shlex.quote(str(data / "convert.log")) + " 2>&1 && python3 -c " +
            shlex.quote("import json,math; from pathlib import Path; d=Path(" + repr(str(data / "lerobot/ds")) +
                "); n=json.loads((d/'meta/info.json').read_text()); assert n['total_episodes']==" + str(count) +
                "; steps=math.ceil(n['total_frames']*5.7/16); Path(" + repr(str(work / "ftq" / (name + ".req"))) +
                ").write_text(str(d)+' '+str(steps)+'\\n'); Path(" + repr(str(work / ("steps_" + name))) +
                ").write_text(str(steps)+'\\n')"))
        subprocess.run(["tmux", "new-session", "-d", "-s", session, converter], check=True)
        print("Converting", name, count, "existing passing demos")
