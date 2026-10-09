"""Stop only the owned round-4 collectors; let existing evaluation children finish."""
import json
import os
from pathlib import Path
import signal
import time

root = (Path.home() / 'kevin').resolve()


def info(pid):
    p = Path('/proc') / str(pid)
    try:
        args = p.joinpath('cmdline').read_bytes().decode().replace('\0', ' ')
        if not p.joinpath('cwd').resolve().is_relative_to(root):
            return None
        env = dict(s.split('=', 1) for s in p.joinpath('environ').read_bytes().decode().split('\0') if '=' in s)
        return args, env.get('TAG')
    except (OSError, ValueError):
        return None


work = root / 'dline/af60v8f57'
for tag in ('af60v8f', 'af60v8f57'):
    w = root / 'dline' / tag
    (w / 'max_round').write_text('3\n')
    (w / 'epoch57_hold').write_text('User stopped chain at round 3; evaluations continue.\n')

# These exact controller PIDs were inspected before this intervention. Pausing
# just the coordinating shell prevents it advancing while its children exit.
controllers = []
for pid in (3045371, 3773416):
    p = info(pid)
    if p and p[1] in ('af60v8f', 'af60v8f57') and ('dagger_line.sh' in p[0] or '_dline.sh' in p[0]):
        os.kill(pid, signal.SIGSTOP)
        controllers.append(pid)
stopped = []
for proc in Path('/proc').glob('[0-9]*'):
    pid = int(proc.name)
    p = info(pid)
    if p and p[1] == 'af60v8f57' and 'collect_sonic_adapter.py' in p[0] and '/datasets/af60v8f57_dagger4' in p[0]:
        os.kill(pid, signal.SIGTERM)
        stopped.append((pid, p))
time.sleep(5)
for pid, identity in stopped:
    if info(pid) == identity:
        os.kill(pid, signal.SIGKILL)
report = dict(time=time.time(), max_round=3, cancelled_collectors=[p for p, _ in stopped],
              paused_coordinators=controllers, data_preserved=True)
(work / 'round_limit_applied.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
# Round-3's eval subshell must remain alive to score and mark completion. Wait
# until it exits before letting the tmux session close with its original shell.
while info(3447653) is not None:
    time.sleep(10)
for pid in controllers:
    p = info(pid)
    if p and p[1] in ('af60v8f', 'af60v8f57'):
        os.kill(pid, signal.SIGKILL)
report.update(coordinators_retired=True, retired_at=time.time())
(work / 'round_limit_applied.json').write_text(json.dumps(report, indent=2))
