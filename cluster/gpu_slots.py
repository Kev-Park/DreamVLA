"""Atomic GPU startup reservations shared by collection and eval launchers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess


def choose_gpus(rows, entries, count, mb, requested_gpu=None, allowed_gpus=None,
                max_gpu_count=None, occupied_gpus=None):
    available, idle = [], []
    occupied = set(occupied_gpus or []) | {e['gpu'] for e in entries if not e.get('resource_policy_block')}
    # Saturated neighbors can use little VRAM. Keep quieter devices first.
    for gpu, used, total, util in sorted(rows, key=lambda row: (row[3] >= 80, row[1])):
        if allowed_gpus is not None and gpu not in allowed_gpus:
            continue
        if requested_gpu is not None and gpu != requested_gpu:
            continue
        pending = sum(e['mb'] for e in entries if e['gpu'] == gpu)
        slots = max(0, (total-2000-used-pending)//mb)
        if slots and max_gpu_count is not None and gpu not in occupied:
            if len(occupied) >= max_gpu_count:
                continue
            occupied.add(gpu)
        available.extend([gpu]*slots)
        if slots and not pending and used < 10000 and util < 20:
            idle.append(gpu)
    # Give parallel workers separate idle GPUs only when another idle GPU
    # remains available. Constrained placements retain the existing packing.
    if requested_gpu is None and len(idle) > count:
        return idle[:count]
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
        policy_path = root/'resource_policy.json'
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        text = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'], text=True)
        fields = [[item.strip() for item in line.split(',')] for line in text.splitlines()]
        by_uuid = {row[1]: int(row[0]) for row in fields}
        rows = [[int(row[0]), *map(int, row[2:])] for row in fields]
        applications = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
        occupied = set()
        for line in applications.splitlines():
            uuid, pid = [item.strip() for item in line.split(',')]
            try:
                if (Path('/proc')/pid).stat().st_uid == os.getuid():
                    occupied.add(by_uuid[uuid])
            except FileNotFoundError:
                pass
        selected = choose_gpus(rows, entries, a.count, a.mb, a.gpu,
                               policy.get('allowed_gpus'), policy.get('max_gpu_count'), occupied)
        entries.extend(dict(owner=a.owner,gpu=gpu,mb=a.mb) for gpu in selected)
        print(' '.join(map(str,selected)))
    temporary = filename.with_suffix('.tmp')
    temporary.write_text(json.dumps(entries))
    temporary.replace(filename)
