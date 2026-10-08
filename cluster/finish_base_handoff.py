"""Finish the already-started Adroit download, verify tensors, launch base eval."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--copy-pid', type=int, required=True)
a = p.parse_args()
def failed(kind, value, trace):
    Path('K:/_ftstage/epoch57/af60v8f57_run01/phase').write_text('failed')
    sys.__excepthook__(kind, value, trace)
sys.excepthook = failed
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
handle = kernel.OpenProcess(0x100000, False, a.copy_pid)
if handle:
    try:
        assert kernel.WaitForSingleObject(handle, 3600000) == 0, 'Download did not finish within one hour'
    finally:
        kernel.CloseHandle(handle)
run = 'af60v8f57_run01'
stage = Path('K:/_ftstage/epoch57')/run
pack = stage/'verified_pack'
manifest = json.loads((pack/'manifest.json').read_text())
assert (pack/'trained.safetensors').stat().st_size > 6*1024**3
assert manifest['packed_keys']
flags = subprocess.CREATE_NO_WINDOW
host = 'sastrygrp-dvij@bluesclues.ist.berkeley.edu'
def call(argv, timeout=1200, cwd=None):
    print('Starting', argv[0], flush=True)
    subprocess.run(argv, check=True, timeout=timeout, creationflags=flags, cwd=cwd)
call(['scp','-q','-r','-o','BatchMode=yes','verified_pack',host+':~/kevin/tmp/af60v8f57_verified_pack'],cwd=stage)
call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,
    '~/kevin/Isaac-GR00T/.venv/bin/python ~/kevin/wt/dagger-watch-sparse/cluster/ckpt_transfer.py unpack '
    '~/kevin/tmp/af60v8f57_verified_pack ~/kevin/checkpoints/af60v7f_run01/checkpoint-10000 '
    '~/kevin/checkpoints/af60v8f57_run01/checkpoint-61000'], timeout=600)
(stage/'phase').write_text('done')
call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,
    'python3 ~/kevin/wt/dagger-watch-sparse/cluster/bootstrap_epoch57.py af60v8f57_run01'],timeout=60)
(stage/'bootstrap.done').touch()
print('Verified base checkpoint delivered; evaluation bootstrapped',flush=True)
