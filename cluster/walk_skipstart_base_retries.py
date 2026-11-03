"""Parallelize future retry batches without changing the fresh-base recipe."""
import argparse
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from account_cpu_usage import sample_account_cpu
from gr00t_budget_chain import ROOT, HERE, atomic, read
from walk_skipstart_chain import FreshChain, TAG


def retry_ids(reports, sweep):
    """Reproduce the resident parent's selection using completed prior sweeps."""
    order=list(range(200));random.Random(42).shuffle(order)
    selected=[]
    ids=order
    for previous in range(sweep):
        prefix='base_initial' if previous==0 else f'base_retry{previous}'
        for k in range(0,len(ids),10):
            name=f'{prefix}_{k//10:03d}'
            if name not in reports:
                return None
            r=reports[name]
            assert r['requested_ids']==ids[k:k+10],name
            assert r['seed']==(0 if previous==0 else 12000+previous),name
            selected.extend(x for x in r['rows'] if x['passing'])
        ids=sorted(set(order)-{r['motion_id'] for r in selected})[:200-len(selected)]
        if not ids:
            return []
    return ids


def assignment(total_batches,current_batch,gpus=(4,6)):
    # Keep the current and next parent batches ahead of peers. Every parent
    # batch before the last contains ten sequential Isaac attempts, so this
    # leaves ample lead for launch and archive-gate installation while avoiding
    # an extra serial ten-reference batch after a late scheduler observation.
    first=max(2,current_batch+2)
    if first>=total_batches or not gpus:
        return None
    return dict(first=first,hold_after=first-1,total=total_batches,
                groups={g:list(range(first+offset,total_batches,len(gpus)))
                        for offset,g in enumerate(gpus)})


def fill_ids(reports, fill):
    """Match the parent's working-reference top-ups after all retry sweeps."""
    order=list(range(200));random.Random(42).shuffle(order)
    selected=[];ids=order
    for previous in range(4):
        prefix='base_initial' if previous==0 else f'base_retry{previous}'
        for k in range(0,len(ids),10):
            name=f'{prefix}_{k//10:03d}'
            if name not in reports:return None
            r=reports[name]
            assert r['requested_ids']==ids[k:k+10],name
            assert r['seed']==(0 if previous==0 else 12000+previous),name
            selected.extend(x for x in r['rows'] if x['passing'])
        ids=sorted(set(order)-{r['motion_id'] for r in selected})[:200-len(selected)]
        if not ids:return []
    ids=sorted({r['motion_id'] for r in selected})[:200-len(selected)]
    for previous in range(1,fill):
        for k in range(0,len(ids),10):
            name=f'base_fill{previous}_{k//10:03d}'
            if name not in reports:return None
            r=reports[name]
            assert r['requested_ids']==ids[k:k+10],name
            assert r['seed']==120000+previous,name
            selected.extend(x for x in r['rows'] if x['passing'])
        ids=sorted({r['motion_id'] for r in selected})[:200-len(selected)]
        if not ids:return []
    return ids


def launch(sweep,is_fill=False):
    stage=f'fill{sweep}' if is_fill else f'retry{sweep}'
    w=ROOT/'dline'/TAG;receipt=w/f'base_{stage}.peers.launch.json'
    if receipt.exists():
        old=read(receipt)
        assert old['status']=='complete','Partial retry launch requires reconciliation'
        return old
    assert not (w/'abort.json').exists()
    if (w/'base_manifest.json').exists():
        return dict(status='base_completed',sweep=sweep)
    reports={p.name.removesuffix('.converted.json'):read(p) for p in w.glob('base_*.converted.json')}
    ids=fill_ids(reports,sweep) if is_fill else retry_ids(reports,sweep)
    if ids is None:
        return dict(status='waiting_for_previous_sweeps',sweep=sweep)
    if not ids:
        return dict(status='no_retry_needed',sweep=sweep)
    prefix=f'base_{stage}'
    current=[int(p.name.split('_')[-1].split('.')[0]) for p in w.glob(prefix+'_*.process.json')]
    plan=assignment((len(ids)+9)//10,max(current,default=-1))
    if plan is None:
        return dict(status='too_small_or_late',sweep=sweep)
    cpu=sample_account_cpu(2);assert cpu['cpu_measurement_complete']
    mask=read(ROOT/'dline/cpu_share.json')['shared_mask'][:12]
    assert len(mask)==12
    projected=cpu['account_cpu_pct']+(len(mask)+1)/os.cpu_count()*100
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    own=set()
    for i,gpu in enumerate(tree.findall('gpu')):
        for proc in gpu.findall('processes/process_info'):
            p=Path('/proc',proc.findtext('pid'))
            if p.exists() and p.stat().st_uid==os.getuid():own.add(i)
    slots=[s for s in read(ROOT/'dline/gpu_slots.json') if Path('/proc',str(s['owner'])).exists()]
    own.update(s['gpu'] for s in slots)
    available=tuple(g for g in (4,6)
                    if not tree.findall('gpu')[g].findall('processes/process_info')
                    and not any(s['gpu']==g for s in slots))
    if not available:
        return dict(status='waiting_for_resources',sweep=sweep,reason='requested GPUs occupied or reserved')
    plan=assignment((len(ids)+9)//10,max(current,default=-1),available)
    groups={gpu:batches for gpu,batches in plan['groups'].items() if batches}
    for batches in groups.values():
        for batch in batches:
            name=f'{prefix}_{batch:03d}'
            assert not (w/(name+'.log')).exists() and not (ROOT/'datasets'/TAG/'raw'/name).exists(),name
    free=shutil.disk_usage(w).free
    if projected>20 or len(own|set(groups)|{0})>5 or free<65_000_000_000:
        return dict(status='waiting_for_resources',sweep=sweep,cpu=cpu,projected_cpu_pct=projected,
                    account_devices=sorted(own),free_bytes=free)
    review=dict(status='launching',sweep=sweep,is_fill=is_fill,prefix=prefix,ids=ids,
                seed=(120000 if is_fill else 12000)+sweep,
                plan=plan,cpu=cpu,mask=mask,account_devices=sorted(own),launched=[],at=time.time())
    atomic(receipt,review)
    for gpu,batches in groups.items():
        name=f'walk_skipstart_{stage}_peer{gpu}_kevin'
        assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
        cmd=[str(ROOT/'wt/locomotion/.venv/bin/python'),'-u',str(Path(__file__)),
             '--fill' if is_fill else '--sweep',str(sweep),'--worker-gpu',str(gpu)]
        shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false\nexec taskset -c '+','.join(map(str,mask))+' '+shlex.join(cmd)+' > '+shlex.quote(str(w/f'{stage}_peer{gpu}.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
        review['launched'].append(dict(gpu=gpu,batches=batches,tmux=name))
        atomic(receipt,review)
    review.update(status='complete',completed_at=time.time());atomic(receipt,review)
    return review


def worker(sweep,gpu,is_fill=False):
    chain=FreshChain();chain.collection_gpu=gpu
    stage=f'fill{sweep}' if is_fill else f'retry{sweep}'
    receipt=read(chain.work/f'base_{stage}.peers.launch.json')
    state=chain.work/(f'retry_peer_fill{sweep}_gpu{gpu}.json' if is_fill else f'retry_peer_s{sweep}_gpu{gpu}.json')
    batches=receipt['plan']['groups'][str(gpu)]
    try:
        for batch in batches:
            atomic(state,dict(status='collecting',pid=os.getpid(),sweep=sweep,batch=batch,at=time.time()))
            ids=receipt['ids'][batch*10:(batch+1)*10]
            chain.batch(f'{receipt["prefix"]}_{batch:03d}',ids,sampled=True,seed=receipt['seed'])
        atomic(state,dict(status='complete',pid=os.getpid(),sweep=sweep,at=time.time()))
    except BaseException as exc:
        atomic(state,dict(status='failed',pid=os.getpid(),sweep=sweep,error=repr(exc),at=time.time()))
        raise


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__)
    stage=p.add_mutually_exclusive_group(required=True)
    stage.add_argument('--sweep',type=int,choices=(1,2,3))
    stage.add_argument('--fill',type=int,choices=range(1,31))
    p.add_argument('--worker-gpu',type=int,choices=(4,6))
    args=p.parse_args()
    number=args.fill if args.fill is not None else args.sweep
    if args.worker_gpu is None:print(json.dumps(launch(number,args.fill is not None)))
    else:worker(number,args.worker_gpu,args.fill is not None)
