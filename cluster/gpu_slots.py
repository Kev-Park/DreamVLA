"""Atomic GPU startup reservations shared by collection and eval launchers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess


def choose_gpus(rows, entries, count, mb, requested_gpu=None):
    available = []
    # Saturated neighbors can use little VRAM. Keep quieter devices first.
    for gpu, used, total, util in sorted(rows, key=lambda row: (row[3] >= 80, row[1])):
        if requested_gpu is not None and gpu != requested_gpu:
            continue
        pending = sum(e['mb'] for e in entries if e['gpu'] == gpu)
        slots = max(0, (total-2000-used-pending)//mb)
        available.extend([gpu]*slots)
    # Pack workers on quiet devices with sufficient memory. Agents review
    # account-wide GPU occupancy before launch; spreading onto every idle GPU
    # can consume the remaining device allowance without improving throughput.
    return available[:count] if len(available) >= min(4, count) else []


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
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'], text=True)
        selected = choose_gpus([[int(x) for x in row.split(',')] for row in rows.splitlines()],
                               entries, a.count, a.mb, a.gpu)
        entries.extend(dict(owner=a.owner,gpu=gpu,mb=a.mb) for gpu in selected)
        print(' '.join(map(str,selected)))
    temporary = filename.with_suffix('.tmp')
    temporary.write_text(json.dumps(entries))
    temporary.replace(filename)
