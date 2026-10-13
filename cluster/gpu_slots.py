"""Atomic GPU startup reservations shared by collection and eval launchers."""
import argparse
import json
import os
from pathlib import Path
import subprocess

def select_slots(rows, entries, count, mb):
    """Use separate GPUs first; share only when fewer GPUs have enough memory."""
    if count < 1 or mb < 1:
        raise ValueError('GPU count and memory request must be positive')
    capacity = []
    for gpu, used, total in sorted(rows, key=lambda row: row[1]):
        pending = sum(e['mb'] for e in entries if e['gpu'] == gpu)
        slots = max(0, (total - 2000 - used - pending) // mb)
        if slots:
            capacity.append((gpu, slots))
    total_slots = sum(slots for _, slots in capacity)
    if total_slots < min(4, count):
        return []
    target = min(count, total_slots)
    selected = [gpu for gpu, _ in capacity][:target]
    if len(selected) < target:
        for gpu, slots in capacity:
            selected.extend([gpu] * min(slots - 1, target - len(selected)))
            if len(selected) == target:
                break
    return selected


def main():
    import fcntl
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
        entries = [e for e in entries if Path('/proc').joinpath(str(e['owner'])).exists()]
        if a.mode == "release":
            for i,e in enumerate(entries):
                if e['owner'] == a.owner and e['gpu'] == a.gpu:
                    entries.pop(i)
                    break
        else:
            output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'], text=True)
            rows = [[int(x) for x in row.split(',')] for row in output.splitlines()]
            selected = select_slots(rows, entries, a.count, a.mb)
            entries.extend(dict(owner=a.owner,gpu=gpu,mb=a.mb) for gpu in selected)
            print(' '.join(map(str,selected)))
        temporary = filename.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries))
        temporary.replace(filename)


if __name__ == '__main__':
    main()
