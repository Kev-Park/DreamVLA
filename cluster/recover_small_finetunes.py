"""Recover the two zero-step GPU-collision failures sequentially, without stopping live jobs."""
import subprocess
import time
from pathlib import Path
import shlex
import json
BL = "sastrygrp-dvij@bluesclues.ist.berkeley.edu"
stage = Path("K:/_ftstage/gr00t_subsets")
for run, steps in (("pick10",1778),("walk10",2154)):
    work = stage/run
    if (work/'phase').read_text().strip() == 'done':
        continue
    command = ("export XLA_PYTHON_CLIENT_PREALLOCATE=false; "
       "g=$(python3 ~/kevin/wt/dagger-watch-sparse/cluster/gpu_slots.py reserve --owner $$ --count 1 --mb 45000); "
       "test -n \"$g\" || exit 2; cd ~/kevin/Isaac-GR00T; "
       "CUDA_VISIBLE_DEVICES=$g .venv/bin/python gr00t/experiment/launch_finetune.py "
       "--base-model-path nvidia/GR00T-N1.7-3B --dataset-path ~/kevin/datasets/"+run+"_base5p7/lerobot/ds "
       "--embodiment-tag unitree_g1_sonic --num-gpus 1 --global-batch-size 16 --dataloader-num-workers 8 "
       "--output-dir ~/kevin/checkpoints --experiment-name "+run+" --max-steps "+str(steps)+
       " --save-steps "+str(steps)+" --save-total-limit 1 --save-only-model > ~/kevin/checkpoints/_ftlogs/"+run+".log 2>&1; "
       "rc=$?; python3 ~/kevin/wt/dagger-watch-sparse/cluster/gpu_slots.py release --owner $$ --gpu $g; "
       "printf '%s\\n' \"$rc\" > ~/kevin/dline/"+run+"/retry.exit")
    check_code = "from pathlib import Path; import sys; matches=[]; exec(\"for p in Path('/proc').glob('[0-9]*'):\\n try:\\n  a=(p/'cmdline').read_bytes().decode().split(chr(0))\\n  if '--experiment-name' in a and a[a.index('--experiment-name')+1]=="+repr(run)+": matches.append(p.name)\\n except (FileNotFoundError,PermissionError): pass\"); sys.exit(3 if matches else 0)"
    preflight = "python3 -c "+shlex.quote(check_code)+" || exit 3; "
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',BL,preflight+
        "tmux new-session -d -s retry_"+run+"_kevin "+shlex.quote(command)],capture_output=True,text=True,timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr or 'Recovery preflight failed')
    (work/'primary').write_text('bluesclues\n')
    (work/'started').write_text(str(int(time.time()))+'\n')
    (work/'recovered_once').write_text('Zero-step collision recovery; GPU reserved atomically\n')
    while True:
        phase=(work/'phase').read_text().strip()
        if phase == 'done': break
        if phase == 'failed': raise RuntimeError(run+' recovery failed; no automatic repeat')
        time.sleep(30)
