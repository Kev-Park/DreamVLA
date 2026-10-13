"""Transfer a directory without compression or local payload storage.

Every file is SHA-256 verified. Inference transfers omit optimizer/RNG state and
reuse matching read-only weight objects already present on the receiving host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

ADDR={'bluesclues':'sastrygrp-dvij@bluesclues.ist.berkeley.edu','biped':'dvij@biped.eecs.berkeley.edu'}
CODE={'bluesclues':'wt/dagger-watch-sparse','biped':'DreamVLA','adroit':'DreamVLA'}


def rooted(relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Expected a path below ~/kevin')
    base=(Path.home()/'kevin').resolve(strict=True)
    out=base/p
    if not out.resolve().is_relative_to(base):raise ValueError('Path escapes ~/kevin')
    return out


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def snapshot(path):
    s=path.stat()
    return {'bytes':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns}


def inventory(directory, transfer_id, inference):
    src=rooted(directory);assert src.is_dir()
    files={}
    for f in sorted(src.rglob('*')):
        if not f.is_file():continue
        if f==src/'stream_manifest.json':continue
        if inference and (f.name in ['optimizer.pt','scheduler.pt'] or f.name.startswith('rng_state')):continue
        if not f.resolve().is_relative_to(rooted('')):raise ValueError('File link escapes ~/kevin')
        before=snapshot(f);digest=sha(f);assert snapshot(f)==before
        files[str(f.relative_to(src))]={'sha256':digest,**before}
    assert files
    m={'source':directory,'files':files,'inference_only':inference,'transfer_id':transfer_id}
    state=rooted('eval_videos/transfers')/transfer_id;state.mkdir(parents=True,exist_ok=False)
    (state/'source.json').write_text(json.dumps(m))
    return m


def prepare(directory, manifest):
    dst=rooted(directory);assert not dst.exists() and not dst.is_symlink(),str(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    stage=dst.parent/('.incoming_'+manifest['transfer_id'])
    stage.mkdir(exist_ok=False)
    store=rooted('checkpoints/_shared_weights/sha256')
    shared={}
    for name,entry in manifest['files'].items():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        if manifest['inference_only'] and name.endswith('.safetensors'):
            candidate=store/(entry['sha256']+'.safetensors')
            if candidate.is_file() and candidate.stat().st_size==entry['bytes']:
                assert sha(candidate)==entry['sha256']
                shared[name]=str(candidate)
    (stage/'stream_manifest.json').write_text(json.dumps(dict(manifest,destination=directory,shared=shared)))
    return {'omitted':list(shared),'stage':str(stage),'destination':directory}


def send(transfer_id, omitted):
    manifest=json.loads((rooted('eval_videos/transfers')/transfer_id/'source.json').read_text())
    src=rooted(manifest['source'])
    assert set(omitted).issubset(manifest['files'])
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',dereference=True) as tar:
        for name,entry in manifest['files'].items():
            if name in omitted:continue
            f=src/name
            assert snapshot(f)=={k:entry[k] for k in ['bytes','inode','mtime_ns']}
            tar.add(f,arcname=name,recursive=False)


def receive(directory, transfer_id):
    dst=rooted(directory);stage=dst.parent/('.incoming_'+transfer_id)
    manifest=json.loads((stage/'stream_manifest.json').read_text());assert manifest['destination']==directory
    expected=set(manifest['files'])-set(manifest['shared']);seen=set()
    with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected and member.name not in seen,member.name
            entry=manifest['files'][member.name];assert member.size==entry['bytes']
            p=stage/member.name;p.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256();total=0
            with tar.extractfile(member) as source,p.open('xb') as target:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    target.write(block);digest.update(block);total+=len(block)
            assert total==entry['bytes'] and digest.hexdigest()==entry['sha256'],member.name
            seen.add(member.name)
    assert seen==expected
    for name,canonical in manifest['shared'].items():
        p=stage/name;p.parent.mkdir(parents=True,exist_ok=True)
        assert Path(canonical).is_file()
        p.symlink_to(os.path.relpath(canonical,p.parent))
    assert not dst.exists()
    stage.rename(dst)
    return {'destination':directory,'verified_files':len(manifest['files']),
            'transmitted_bytes':sum(manifest['files'][n]['bytes'] for n in seen),
            'reused_weight_files':list(manifest['shared']),'transfer_id':transfer_id,'time':time.time()}


def ssh(host, command):
    if host=='adroit':
        return ['wsl.exe','-d','Ubuntu','-u','kevy0','--','bash','-lc',
                'ssh -o ControlPath=~/.ssh/cm-adroit -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15 adroit '+shlex.quote(command)]
    return [r'C:\Windows\System32\OpenSSH\ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=15',ADDR[host],command]


def remote_command(host, *args):
    return 'python3 ~/kevin/'+CODE[host]+'/cluster/stream_directory.py '+' '.join(shlex.quote(str(a)) for a in args)


def remote_json(host, args, payload=None, timeout=300):
    result=subprocess.run(ssh(host,remote_command(host,*args)),input=json.dumps(payload) if payload is not None else None,
                          capture_output=True,text=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError(host+': '+result.stderr[-4000:])
    return json.loads(result.stdout)


def transfer(args):
    ident=uuid.uuid4().hex
    src=args.source_host;dst=args.destination_host
    inv_args=['inventory',args.source,ident]
    if args.inference:inv_args.append('--inference')
    manifest=remote_json(src,inv_args)
    prep=remote_json(dst,['prepare',args.destination],manifest)
    options=dict(creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    sender=subprocess.Popen(ssh(src,remote_command(src,'send',ident,'--omitted',json.dumps(prep['omitted']))),
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,**options)
    receiver=subprocess.Popen(ssh(dst,remote_command(dst,'receive',args.destination,ident)),
                              stdin=sender.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**options)
    sender.stdout.close()
    print(json.dumps({'state':'streaming','source_host':src,'destination_host':dst,'transfer_id':ident,
                     'files':len(manifest['files']),'reused':prep['omitted'],
                     'bytes':sum(v['bytes'] for k,v in manifest['files'].items() if k not in prep['omitted'])}),flush=True)
    output,errors=receiver.communicate(timeout=7200)
    source_errors=sender.stderr.read();source_code=sender.wait(timeout=30)
    if receiver.returncode or source_code:
        raise RuntimeError(f'Transfer failed: source={source_code} receiver={receiver.returncode}; '+
                           source_errors.decode(errors='replace')[-2000:]+errors.decode(errors='replace')[-2000:])
    receipt=json.loads(output);receipt.update(source_host=src,destination_host=dst,source=args.source)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('inventory');a.add_argument('directory');a.add_argument('transfer_id');a.add_argument('--inference',action='store_true')
    a=s.add_parser('prepare');a.add_argument('directory')
    a=s.add_parser('send');a.add_argument('transfer_id');a.add_argument('--omitted',default='[]')
    a=s.add_parser('receive');a.add_argument('directory');a.add_argument('transfer_id')
    a=s.add_parser('transfer');a.add_argument('--source-host',choices=[*ADDR,'adroit'],required=True)
    a.add_argument('--destination-host',choices=[*ADDR,'adroit'],required=True);a.add_argument('--source',required=True)
    a.add_argument('--destination',required=True);a.add_argument('--inference',action='store_true');a.add_argument('--receipt',type=Path)
    args=p.parse_args()
    if args.mode=='inventory':result=inventory(args.directory,args.transfer_id,args.inference)
    elif args.mode=='prepare':result=prepare(args.directory,json.load(sys.stdin))
    elif args.mode=='send':send(args.transfer_id,json.loads(args.omitted));return
    elif args.mode=='receive':result=receive(args.directory,args.transfer_id)
    else:transfer(args);return
    print(json.dumps(result))


if __name__=='__main__':main()
