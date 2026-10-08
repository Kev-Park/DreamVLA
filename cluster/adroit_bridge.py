"""Bounded, hidden WSL SSH bridge for Git Bash orchestrators."""
import subprocess
import sys

try:
    result = subprocess.run(
        ['wsl.exe', '-e', 'bash', '-lc',
         'ssh -o ControlPath=~/.ssh/adroit.sock -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=20 adroit bash -s'],
        input=sys.stdin.buffer.read(), capture_output=True, timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)
except subprocess.TimeoutExpired:
    sys.stderr.write('Adroit command timed out; connection requires review, no automatic retry.\n')
    raise SystemExit(124)
