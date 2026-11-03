"""Task-scoped preflight for future parallel walking DAgger collection.

Never changes or restarts an active collector. Existing account jobs are only
observed; unavailable resources defer this task rather than modifying them.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from account_cpu_usage import sample_account_cpu
from gr00t_budget_chain import ROOT, HERE, atomic, read, checkpoint_ok
from walk_skipstart_chain import TAG


def assignment(current_batch,extra_peer=False,devices=None):
    if devices is None:
        devices=(4,6,7) if extra_peer else (4,6)
    if not devices:
        return None
    # Actual expert collection measured ~90 s/100 steps on GPU0 versus
    # ~28 s on idle peers. Give the slower parent fewer complete batches;
    # preserve the same three-batch launch lead and canonical batch seeds.
    effective_workers=1+3*len(devices)
    start=max((20+effective_workers-1)//effective_workers,current_batch+3)
    if start>=20:
        return None
    return dict(first=start,hold_after=start-1,
                groups={gpu:list(range(start+offset,20,len(devices)))
                        for offset,gpu in enumerate(devices)})


def launch(round_id):
    w=ROOT/'dline'/TAG
    receipt=w/f'r{round_id}.peers.launch.json'
    if receipt.exists():
        prior=read(receipt)
        assert prior['status']=='complete','Partial launch requires reconciliation, not retry'
        return prior
    trained=w/f'{TAG}_r{round_id-1}.trained.json'
    if not trained.exists():
        return dict(status='waiting_for_checkpoint',round=round_id)
    assert not (w/'abort.json').exists(),'Main controller has failed; review before any new work'
    checkpoint=Path(read(trained)['checkpoint']);assert checkpoint_ok(checkpoint)
    if (w/f'collection_r{round_id}.json').exists():
        return dict(status='already_collected',round=round_id)
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    free=shutil.disk_usage(w).free
    own=set()
    for i,node in enumerate(tree.findall('gpu')):
        for p in node.findall('processes/process_info'):
            q=Path('/proc',p.findtext('pid'))
            if q.exists() and q.stat().st_uid==os.getuid():own.add(i)
    slots=[s for s in read(ROOT/'dline/gpu_slots.json') if Path('/proc',str(s['owner'])).exists()]
    own.update(s['gpu'] for s in slots)
    devices=[]
    limit=3 if free>=80_000_000_000 else 2
    for gpu in (4,6,7,9):
        if (not tree.findall('gpu')[gpu].findall('processes/process_info')
            and not any(s['gpu']==gpu for s in slots)
            and len(own|{0,5}|set(devices)|{gpu})<=5):
            devices.append(gpu)
        if len(devices)==limit:break
    if not devices:
        return dict(status='waiting_for_resources',round=round_id,reason='No idle unreserved peer GPU within account cap')
    extra=len(devices)==3
    indices=[int(p.name.split('batch')[1].split('.')[0]) for p in w.glob(f'r{round_id}_batch*.process.json')]
    plan=assignment(max(indices,default=-1),devices=devices)
    if plan is None:
        return dict(status='too_late_for_peers',round=round_id)
    groups={gpu:ids for gpu,ids in plan['groups'].items() if ids}
    for ids in groups.values():
        for batch in ids:
            label=f'r{round_id}_batch{batch:03d}'
            assert not (w/(label+'.log')).exists(),'Future batch already attempted; reconcile first'
            assert not (ROOT/'datasets'/TAG/'raw'/label).exists()
    cpu=sample_account_cpu(2)
    assert cpu['cpu_measurement_complete']
    mask=read(ROOT/'dline/cpu_share.json')['shared_mask'][:12]
    assert len(mask)==12
    projected=cpu['account_cpu_pct']+(len(mask)+1)/os.cpu_count()*100
    if projected>20:
        return dict(status='waiting_for_resources',round=round_id,cpu=cpu,projected_cpu_pct=projected)
    if len(own|set(groups)|{0,5})>5 or free<47_000_000_000:
        return dict(status='waiting_for_resources',round=round_id,account_devices=sorted(own),
                    free_bytes=free)
    for gpu in groups:
        if tree.findall('gpu')[gpu].findall('processes/process_info') or any(s['gpu']==gpu for s in slots):
            return dict(status='waiting_for_resources',round=round_id,occupied_requested_gpu=gpu)
    review=dict(status='launching',round=round_id,checkpoint=str(checkpoint),plan=plan,
                cpu=cpu,mask=mask,account_devices=sorted(own),free_bytes=free,
                extra_collection_peer=extra,parent_relative_batch_cost=3,
                at=time.time(),launched=[])
    atomic(receipt,review)
    for gpu,batches in groups.items():
        name=f'walk_skipstart_r{round_id}_peer{gpu}_kevin'
        assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
        cmd=[str(ROOT/'wt/locomotion/.venv/bin/python'),'-u',str(HERE/'walk_skipstart_chain.py'),
             '--collection-gpu',str(gpu),'--dagger-peer-round',str(round_id),
             '--dagger-peer-batches',*map(str,batches)]
        shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false\nexec taskset -c '+','.join(map(str,mask))+' '+shlex.join(cmd)+' > '+shlex.quote(str(w/f'r{round_id}_peer{gpu}.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
        review['launched'].append(dict(tmux=name,gpu=gpu,batches=batches,command=cmd))
        atomic(receipt,review)
    review.update(status='complete',completed_at=time.time())
    atomic(receipt,review)
    return review


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round',type=int,choices=(1,2,3),required=True)
    print(json.dumps(launch(p.parse_args().round)))
