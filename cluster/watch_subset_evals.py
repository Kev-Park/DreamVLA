"""Launch subset evals as delivered checkpoints become available; stop on SSH errors."""
from pathlib import Path
import subprocess
import time
stage=Path('K:/_ftstage/gr00t_subsets')
while True:
    remaining=0
    for name in ('pick10','pick50','walk10','walk50'):
        work=stage/name
        if (work/'eval.started').exists(): continue
        remaining+=1
        if (work/'phase').read_text().strip()!='done': continue
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','sastrygrp-dvij@bluesclues.ist.berkeley.edu',
            'python3 ~/kevin/wt/dagger-watch-sparse/cluster/bootstrap_subset_eval.py '+name],capture_output=True,text=True,timeout=45,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if p.returncode: raise RuntimeError(p.stderr or p.stdout)
        (work/'eval.started').write_text(str(time.time()))
    if not remaining: break
    time.sleep(30)
