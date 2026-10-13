"""Atomic GPU startup reservations shared by collection and eval launchers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["reserve", "release"])
p.add_argument("--owner", type=int, required=True)
p.add_argument("--count", type=int, default=4)
p.add_argument("--mb", type=int, default=15500)
p.add_argument("--gpu", type=int)
a = p.parse_args()
root = Path.home() / "kevin/dline"
with (root / "gpu_slots.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    filename = root / "gpu_slots.json"
    entries = json.loads(filename.read_text()) if filename.exists() else []
    entries = [e for e in entries if Path('/proc') .joinpath(str(e['owner'])).exists()]
    if a.mode == "release":
        for i,e in enumerate(entries):
            if e['owner'] == a.owner and e['gpu'] == a.gpu:
                entries.pop(i)
                break
    else:
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'], text=True)
        available = []
        for gpu,used,total in sorted(([int(x) for x in row.split(',')] for row in rows.splitlines()), key=lambda row:row[1]):
            pending = sum(e['mb'] for e in entries if e['gpu']==gpu)
            available.extend([gpu] * max(0,(total-2000-used-pending)//a.mb))
        selected = available[:a.count] if len(available)>=min(4,a.count) else []
        entries.extend(dict(owner=a.owner,gpu=gpu,mb=a.mb) for gpu in selected)
        print(' '.join(map(str,selected)))
    temporary = filename.with_suffix('.tmp')
    temporary.write_text(json.dumps(entries))
    temporary.replace(filename)
