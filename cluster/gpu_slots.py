"""Atomic GPU startup reservations shared by collection and eval launchers."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

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


def reservation_paused(pauses, owner, owner_start, count, mb):
    return any(p['owner']==owner and str(p['owner_start'])==str(owner_start)
               and p['count']==count and p['mb']==mb for p in pauses)


def reviewed_placement(placements, owner, owner_start, count, mb):
    """Read an explicit placement for one manually reviewed job generation."""
    matches = [p for p in placements if reservation_paused([p], owner, owner_start, count, mb)]
    if len(matches) > 1:
        raise ValueError('Multiple placements for the same job request')
    if not matches:
        return None
    gpus = matches[0]['gpus']
    if len(gpus) != count or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('Explicit placement must name one GPU per requested worker')
    return list(gpus)


def placed_slots(rows, entries, gpus, mb):
    """Honor a job's explicit devices while retaining memory reservations."""
    capacity = {gpu: total - 2000 - used for gpu, used, total in rows}
    if not set(gpus) <= capacity.keys():
        raise ValueError('Explicit placement names an unavailable GPU')
    for gpu, workers in Counter(gpus).items():
        pending = sum(e['mb'] for e in entries if e['gpu'] == gpu)
        if pending + workers * mb > capacity[gpu]:
            return []
    return list(gpus)


def observed_reserved_memory(processes, parents, entries):
    """Credit each allocation only to its nearest reserving ancestor."""
    owners = {(e['gpu'], e['owner']) for e in entries}
    allocations = {}
    for gpu, pid, used in processes:
        allocations[gpu, pid] = max(used, allocations.get((gpu, pid), 0))
    observed = {}
    for (gpu, pid), used in allocations.items():
        seen = set()
        while pid and pid not in seen:
            seen.add(pid)
            if (gpu, pid) in owners:
                observed[gpu, pid] = observed.get((gpu, pid), 0) + used
                break
            pid = parents.get(pid, 0)
    return observed

def pending_reserved_mb(gpu, entries, observed):
    targets = {}
    for entry in entries:
        if entry['gpu'] == gpu:
            targets[entry['owner']] = targets.get(entry['owner'], 0) + entry['mb']
    # Device used memory already includes these live allocations. Keep each
    # owner's remaining headroom; unknown ancestry retains the full reservation.
    return sum(max(0, target - observed.get((gpu, owner), 0))
               for owner, target in targets.items())

def choose_gpus(rows, entries, count, mb, requested_gpu=None, observed=None):
    available = []
    # Saturated neighbors can use little VRAM. Keep quieter devices first.
    for gpu, used, total, util in sorted(rows, key=lambda row: (row[3] >= 80, row[1])):
        if requested_gpu is not None and gpu != requested_gpu:
            continue
        pending = pending_reserved_mb(gpu, entries, observed or {})
        slots = max(0, (total-2000-used-pending)//mb)
        available.extend([gpu]*slots)
    # Pack workers on quiet devices with sufficient memory. Agents review
    # account-wide GPU occupancy before launch; spreading onto every idle GPU
    # can consume the remaining device allowance without improving throughput.
    return available[:count] if len(available) >= min(4, count) else []

def memory_snapshot(entries):
    tree = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15))
    rows, processes = [], []
    for gpu, node in enumerate(tree.findall('gpu')):
        rows.append([gpu, int(node.findtext('fb_memory_usage/used').split()[0]),
                     int(node.findtext('fb_memory_usage/total').split()[0]),
                     int(node.findtext('utilization/gpu_util').split()[0])])
        for process in node.findall('processes/process_info'):
            memory = process.findtext('used_memory', '').split()
            if memory and memory[0].isdigit():
                processes.append((gpu, int(process.findtext('pid')), int(memory[0])))
    parents = {}
    for path in Path('/proc').iterdir():
        try:
            if path.name.isdigit() and path.stat().st_uid == os.getuid():
                parents[int(path.name)] = int((path/'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            pass
    return rows, observed_reserved_memory(processes, parents, entries)


def remaining_reservations(entries, observed):
    """Credit each live allocation once while retaining unconsumed headroom."""
    credits = dict(observed)
    pending = []
    for entry in entries:
        key = (entry['gpu'], entry['owner'])
        used = min(entry['mb'], credits.get(key, 0))
        credits[key] = credits.get(key, 0) - used
        pending.append(dict(entry, mb=entry['mb'] - used))
    return pending


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
            pause_file=root/'gpu_reservation_pauses.json'
            pauses=json.loads(pause_file.read_text()) if pause_file.exists() else []
            owner_start=Path(f'/proc/{a.owner}/stat').read_text().rsplit(')',1)[1].split()[19]
            if reservation_paused(pauses,a.owner,owner_start,a.count,a.mb):
                selected=[]
            else:
                rows, observed = memory_snapshot(entries)
                memory_rows = [row[:3] for row in rows]
                pending = remaining_reservations(entries, observed)
                placement_file = root/'gpu_reservation_placements.json'
                placements = json.loads(placement_file.read_text()) if placement_file.exists() else []
                requested = reviewed_placement(placements, a.owner, owner_start, a.count, a.mb)
                if requested is not None:
                    selected = placed_slots(memory_rows, pending, requested, a.mb)
                elif a.gpu is not None:
                    selected = choose_gpus(rows, entries, a.count, a.mb, a.gpu, observed)
                else:
                    selected = select_slots(memory_rows, pending, a.count, a.mb)
            entries.extend(dict(owner=a.owner,gpu=gpu,mb=a.mb) for gpu in selected)
            print(' '.join(map(str,selected)))
        temporary = filename.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries))
        temporary.replace(filename)


if __name__ == '__main__':
    main()
