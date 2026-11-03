"""Precomplete the canonical second top-up while its parent awaits an archive."""
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
from gr00t_budget_chain import ROOT, atomic, read
from walk_skipstart_base_retries import fill_ids
from walk_skipstart_chain import FreshChain, TAG


def launch():
    w=ROOT/'dline'/TAG; receipt=w/'small_fill2.launch.json'
    if receipt.exists():
        old=read(receipt)
        assert old['status']=='complete','Partial launch requires review'
        return old
    assert not (w/'abort.json').exists()
    reports={p.name.removesuffix('.converted.json'):read(p)
             for p in w.glob('base_*.converted.json')}
    ids=fill_ids(reports,2)
    if ids is None:return dict(status='waiting_for_previous_sweeps')
    if not ids:return dict(status='no_fill_needed')
    assert len(ids)<=10,ids
    # The local archive fence must retain this acknowledgement until our
    # converted receipt exists. Its absence proves the parent cannot advance.
    assert (w/'base_fill1_001.converted.json').exists()
    assert not (w/'base_fill1_001.archived.json').exists(),'Parent already released'
    label='base_fill2_000'
    assert not any((w/(label+s)).exists() for s in ('.log','.process.json','.converted.json'))
    assert not (ROOT/'datasets'/TAG/'raw'/label).exists()
    cpu=sample_account_cpu(2); assert cpu['cpu_measurement_complete']
    mask=read(ROOT/'dline/cpu_share.json')['shared_mask'][:12]
    assert len(mask)==12
    projected=cpu['account_cpu_pct']+13/os.cpu_count()*100
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    own=set()
    for index,gpu in enumerate(tree.findall('gpu')):
        for process in gpu.findall('processes/process_info'):
            proc=Path('/proc',process.findtext('pid'))
            if proc.exists() and proc.stat().st_uid==os.getuid():own.add(index)
    slots=[s for s in read(ROOT/'dline/gpu_slots.json') if Path('/proc',str(s['owner'])).exists()]
    own.update(s['gpu'] for s in slots)
    available=[g for g in (4,7,6,9)
               if not tree.findall('gpu')[g].findall('processes/process_info')
               and not any(s['gpu']==g for s in slots) and len(own|{0,g})<=5]
    free=shutil.disk_usage(w).free
    if projected>20 or free<65_000_000_000 or not available:
        return dict(status='waiting_for_resources',cpu=cpu,projected_cpu_pct=projected,
                    free_bytes=free,account_devices=sorted(own))
    gpu=available[0]; name='walk_skipstart_smallfill2_kevin'
    assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
    review=dict(status='launching',ids=ids,seed=120002,gpu=gpu,mask=mask,
                cpu=cpu,free_bytes=free,account_devices=sorted(own),at=time.time())
    atomic(receipt,review)
    cmd=[str(ROOT/'wt/locomotion/.venv/bin/python'),'-u',str(Path(__file__)),'--worker']
    shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false\nexec taskset -c '+','.join(map(str,mask))+' '+shlex.join(cmd)+' > '+shlex.quote(str(w/'small_fill2.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
    review.update(status='complete',launched_at=time.time());atomic(receipt,review)
    return review


def worker():
    chain=FreshChain(); receipt=read(chain.work/'small_fill2.launch.json')
    chain.collection_gpu=receipt['gpu']
    state=chain.work/f'retry_peer_smallfill2_gpu{chain.collection_gpu}.json'
    try:
        atomic(state,dict(status='collecting',pid=os.getpid(),batch=0,at=time.time()))
        chain.batch('base_fill2_000',receipt['ids'],sampled=True,seed=120002)
        atomic(state,dict(status='complete',pid=os.getpid(),at=time.time()))
    except BaseException as exc:
        atomic(state,dict(status='failed',pid=os.getpid(),error=repr(exc),at=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    if p.parse_args().worker:worker()
    else:print(json.dumps(launch()))
