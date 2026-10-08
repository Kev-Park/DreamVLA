"""Upgrade the first pick collection driver only at its completed-pass boundary."""
import os
from pathlib import Path
import signal
import subprocess
import time
root=Path.home()/'kevin'
pid=699995
work=root/'dline/expertpick1200'
while Path(f'/proc/{pid}').exists():
    ready=[p for p in work.glob('pass*.ready') if not p.with_suffix('.archived').exists()]
    if ready:
        active=False
        for p in Path('/proc').glob('[0-9]*'):
            try:
                a=(p/'cmdline').read_bytes().decode().split(chr(0))
                if any(x.endswith('/collect_sonic_adapter.py') for x in a) and '--output-directory' in a:
                    active |= '/expertpick1200_pass' in a[a.index('--output-directory')+1]
            except (FileNotFoundError,PermissionError): pass
        if not active:
            proc=Path(f'/proc/{pid}')
            env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split(chr(0)) if '=' in x)
            assert env.get('TAG')=='expertpick1200' and proc.stat().st_uid==os.getuid()
            os.kill(pid,signal.SIGTERM)
            for _ in range(20):
                if not proc.exists(): break
                time.sleep(.5)
            if proc.exists(): raise RuntimeError('Waiting driver did not stop; no forced kill')
            subprocess.run(['tmux','kill-session','-t','expertpick1200_kevin'],capture_output=True)
            subprocess.run(['python3',str(root/'wt/dagger-watch-sparse/cluster/launch_expert1200.py')],check=True)
            print('Pick collection upgraded at completed-pass boundary',flush=True)
            break
    time.sleep(15)
