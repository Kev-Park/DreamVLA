"""Cap Kevin's bluesclues workers collectively through a shared CPU mask."""
import json
import os
from pathlib import Path
import time

root = (Path.home() / 'kevin').resolve()
available = sorted(os.sched_getaffinity(0))
count = max(1, int(len(available) * 0.30))
mask = {available[i * len(available) // count] for i in range(count)}
status = root / 'dline/cpu_share.json'
while True:
    changed = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if (proc / 'status').stat().st_uid != os.getuid():
                continue
            cwd = (proc / 'cwd').resolve()
            if not cwd.is_relative_to(root):
                continue
            args = (proc / 'cmdline').read_bytes()
            if b'tmux' in args.split(b'\0')[0]:
                continue
            for thread in (proc / 'task').iterdir():
                os.sched_setaffinity(int(thread.name), mask)
            changed.append(int(proc.name))
        except (OSError, ValueError):
            continue
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({'time': time.time(), 'available_cpus': len(available),
                                 'shared_mask': sorted(mask), 'pids': changed}))
    time.sleep(10)
