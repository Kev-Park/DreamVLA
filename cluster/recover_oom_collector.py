"""Re-run an owned OOM collector while retaining its parent's completion barrier."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('pid', type=int)
p.add_argument('gpu', type=int)
a = p.parse_args()
proc = Path('/proc') / str(a.pid)
root = (Path.home() / 'kevin').resolve()
cwd = (proc / 'cwd').resolve()
assert cwd.is_relative_to(root)
args = [s for s in (proc / 'cmdline').read_bytes().decode().split('\0') if s]
assert len(args) > 1 and args[1].endswith('/collect_sonic_adapter.py')
log = (proc / 'fd/1').resolve()
assert log.is_relative_to(root) and 'torch.OutOfMemoryError' in log.read_text(errors='replace')
assert '[INFO] motion ' not in log.read_text(errors='replace'), 'Only recover failed startup; active rollouts require review'
assert '--skip-existing' in args
env = dict(s.split('=', 1) for s in (proc / 'environ').read_bytes().decode().split('\0') if '=' in s)
env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
args[0] = str(proc / 'exe')
status = log.with_suffix('.recovery.json')
recovery_log = log.with_suffix('.recovery.log')
assert not status.exists(), 'Recovery already attempted; inspect its status first'
state = dict(original_pid=a.pid, gpu=a.gpu, started=time.time(), state='starting', log=str(recovery_log))
status.write_text(json.dumps(state, indent=2))
# The failed Isaac shutdown can spin indefinitely. Keep its PID as the original
# shell's wait barrier so conversion cannot race this replacement worker.
os.kill(a.pid, signal.SIGSTOP)
try:
    with recovery_log.open('wb') as out:
        child = subprocess.Popen(args, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
        state.update(state='running', replacement_pid=child.pid)
        status.write_text(json.dumps(state, indent=2))
        rc = child.wait()
    state.update(state='done' if rc == 0 else 'failed', returncode=rc, finished=time.time())
    status.write_text(json.dumps(state, indent=2))
    if rc == 0:
        os.kill(a.pid, signal.SIGKILL)
except Exception as error:
    state.update(state='failed', error=str(error), finished=time.time())
    status.write_text(json.dumps(state, indent=2))
    raise
