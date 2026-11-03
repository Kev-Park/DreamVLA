"""Precollect one provably unchanged top-up batch while final retries run."""
import argparse
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from account_cpu_usage import sample_account_cpu
from gr00t_budget_chain import ROOT, atomic, read
from walk_skipstart_chain import FreshChain, TAG


def stable_first_fill_batch(reports, size=10):
    order=list(range(200));random.Random(42).shuffle(order)
    selected=[];unknown=[];missing=[];ids=order
    for sweep in range(4):
        prefix='base_initial' if sweep==0 else f'base_retry{sweep}'
        for offset in range(0,len(ids),10):
            name=f'{prefix}_{offset//10:03d}';chunk=ids[offset:offset+10]
            if name not in reports:
                if sweep<3:return None
                unknown.extend(chunk);missing.append(name)
                continue
            r=reports[name]
            assert r['requested_ids']==chunk and r['seed']==(0 if sweep==0 else 12000+sweep),name
            selected.extend(x['motion_id'] for x in r['rows'] if x['passing'])
        ids=sorted(set(order)-set(selected))[:200-len(selected)]
    assert len(set(selected))==len(selected),'Pre-fill successes must be unique'
    minimum=200-len(selected)-len(unknown)
    working=sorted(selected)[:size]
    # Even if every unknown attempt passes, these attempts are still needed.
    # Unknown references cannot insert before this prefix of working IDs.
    if len(unknown)<20 or minimum<size or len(working)<size or max(working)>=min(unknown):
        return None
    return dict(ids=working,seed=120001,guaranteed_remaining=minimum,
                unresolved_ids=unknown,unresolved_batches=missing)


def launch():
    w=ROOT/'dline'/TAG;receipt=w/'early_fill1.launch.json'
    if receipt.exists():
        old=read(receipt);assert old['status']=='complete','Partial launch requires review'
        return old
    assert not (w/'abort.json').exists()
    reports={p.name.removesuffix('.converted.json'):read(p) for p in w.glob('base_*.converted.json')}
    proof=stable_first_fill_batch(reports)
    if proof is None:return dict(status='no_proven_overlap')
    if not any(not (w/(name+'.process.json')).exists() for name in proof['unresolved_batches']):
        return dict(status='insufficient_launch_lead')
    label='base_fill1_000'
    if any((w/(label+suffix)).exists() for suffix in ('.log','.process.json','.converted.json')):
        return dict(status='already_started')
    assert not (ROOT/'datasets'/TAG/'raw'/label).exists()
    cpu=sample_account_cpu(2);assert cpu['cpu_measurement_complete']
    mask=read(ROOT/'dline/cpu_share.json')['shared_mask'][:12];assert len(mask)==12
    projected=cpu['account_cpu_pct']+(len(mask)+1)/os.cpu_count()*100
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    own=set()
    for i,gpu in enumerate(tree.findall('gpu')):
        for process in gpu.findall('processes/process_info'):
            p=Path('/proc',process.findtext('pid'))
            if p.exists() and p.stat().st_uid==os.getuid():own.add(i)
    slots=[s for s in read(ROOT/'dline/gpu_slots.json') if Path('/proc',str(s['owner'])).exists()]
    own.update(s['gpu'] for s in slots)
    free=shutil.disk_usage(w).free
    if (projected>20 or len(own|{0,9})>5 or free<65_000_000_000
        or tree.findall('gpu')[9].findall('processes/process_info')
        or any(s['gpu']==9 for s in slots)):
        return dict(status='waiting_for_resources',cpu=cpu,projected_cpu_pct=projected,
                    account_devices=sorted(own),free_bytes=free)
    name='walk_skipstart_earlyfill_kevin'
    assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
    review=dict(status='launching',proof=proof,cpu=cpu,mask=mask,gpu=9,
                account_devices=sorted(own),free_bytes=free,tmux=name,at=time.time())
    atomic(receipt,review)
    cmd=[str(ROOT/'wt/locomotion/.venv/bin/python'),'-u',str(Path(__file__)),'--worker']
    shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false\nexec taskset -c '+','.join(map(str,mask))+' '+shlex.join(cmd)+' > '+shlex.quote(str(w/'early_fill1.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
    review.update(status='complete',launched_at=time.time());atomic(receipt,review)
    return review


def worker():
    chain=FreshChain();chain.collection_gpu=9
    proof=read(chain.work/'early_fill1.launch.json')['proof']
    state=chain.work/'retry_peer_earlyfill1_gpu9.json'
    try:
        atomic(state,dict(status='collecting',pid=os.getpid(),batch=0,at=time.time()))
        chain.batch('base_fill1_000',proof['ids'],sampled=True,seed=proof['seed'])
        atomic(state,dict(status='complete',pid=os.getpid(),at=time.time()))
    except BaseException as error:
        atomic(state,dict(status='failed',pid=os.getpid(),error=repr(error),at=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    if parser.parse_args().worker:worker()
    else:print(json.dumps(launch()))
