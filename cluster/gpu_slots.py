"""Atomic GPU startup reservations shared by collection and eval launchers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


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
        entries = [e for e in entries if Path('/proc') .joinpath(str(e['owner'])).exists()]
        if a.mode == "release":
            for i,e in enumerate(entries):
                if e['owner'] == a.owner and e['gpu'] == a.gpu:
                    entries.pop(i)
                    break
        else:
            rows, observed = memory_snapshot(entries)
            selected = choose_gpus(rows, entries, a.count, a.mb, a.gpu, observed)
            entries.extend(dict(owner=a.owner,gpu=gpu,mb=a.mb) for gpu in selected)
            print(' '.join(map(str,selected)))
        temporary = filename.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries))
        temporary.replace(filename)


if __name__ == '__main__':
    main()
