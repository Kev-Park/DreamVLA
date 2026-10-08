"""Bounded, hidden WSL SSH bridge for Git Bash orchestrators."""
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--timeout', type=int, default=45)
args = parser.parse_args()

try:
    result = subprocess.run(
        ['wsl.exe', '-e', 'bash', '-lc',
         'ssh -o ControlPath=~/.ssh/adroit.sock -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=20 adroit bash -s'],
        input=sys.stdin.buffer.read(), capture_output=True, timeout=args.timeout,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)
except subprocess.TimeoutExpired:
    sys.stderr.write('Adroit remote command exceeded its time limit; inspect the command before retrying.\n')
    raise SystemExit(124)
