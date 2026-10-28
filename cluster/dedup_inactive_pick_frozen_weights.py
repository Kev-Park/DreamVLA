"""Maintenance only: fully verify inactive weight copies before atomic sharing."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time


def metadata(path):
    s=path.stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,
                nlink=s.st_nlink,allocated_bytes=s.st_blocks*512)


def durable(path,data):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        f.write(json.dumps(data,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def digest(path):
    before=metadata(path);h=hashlib.sha256()
    with path.open('rb') as f:
        assert os.fstat(f.fileno()).st_ino==before['inode']
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert metadata(path)==before,'File changed while hashing'
    return h.hexdigest()


def account_fraction():
    group='/user.slice/user-%d.slice'%os.getuid()
    counter=Path('/sys/fs/cgroup')/group.lstrip('/')/'cpu.stat'
    def read():
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if p.stat().st_uid!=os.getuid():continue
                value=(p/'cgroup').read_text().strip()
                assert value=='0::'+group or value.startswith('0::'+group+'/')
            except FileNotFoundError:continue
        return int(dict(line.split() for line in counter.read_text().splitlines())['usage_usec'])
    before=read();start=time.monotonic();time.sleep(2);after=read()
    return (after-before)/1e6/(time.monotonic()-start)/os.cpu_count()


def main(work):
    root=(Path.home()/'kevin').resolve();request=json.loads((work/'request.json').read_text())
    assert request['cpu_limit']==.20 and request['shared_mask']==[44]
    os.sched_setaffinity(0,{44})
    target=root/request['target'];expected=request['expected_sha256']
    assert target.parent==root/'checkpoints/_shared_weights/sha256'
    assert target.name==expected+'.safetensors' and not target.is_symlink()
    assert metadata(target)==request['target_metadata']
    ledger=work/'verified_dedup.jsonl';pending=work/'pending.json'
    assert not ledger.exists() and not pending.exists(),'Reconcile any existing transaction first'
    completed=reclaimed=0
    def status(phase,**extra):
        durable(work/'status.json',dict(time=time.time(),pid=os.getpid(),phase=phase,
                completed_files=completed,reclaimed_bytes=reclaimed,total_files=len(request['pairs']),**extra))
    def admission():
        while True:
            usage=account_fraction()
            if usage<=.19:return
            status('waiting_cpu_headroom',account_cpu_fraction=usage);time.sleep(45)
    def validate(row):
        source=root/row['source']
        assert source.parent==root/row['checkpoint']
        assert source.parent.parent.name==row['run']
        assert source.name=='model-00001-of-00003.safetensors'
        assert source.resolve(strict=True)==source and not source.is_symlink()
        assert source.resolve().is_relative_to(root/'checkpoints')
        assert stat.S_ISREG(source.stat().st_mode) and source.stat().st_nlink==1
        assert metadata(source)==row['metadata']
        assert metadata(target)==request['target_metadata']
        index=source.parent/'model.safetensors.index.json'
        assert hashlib.sha256(index.read_bytes()).hexdigest()==row['index_sha256']
        for shard in set(json.loads(index.read_text())['weight_map'].values()):
            assert Path(shard).name==shard and (source.parent/shard).is_file()
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if p.stat().st_uid!=os.getuid():continue
                command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
                assert str(source.parent) not in command,'Checkpoint acquired a live user'
            except FileNotFoundError:continue
        return source
    try:
        status('ready');admission();status('checksumming_canonical',current=request['target'])
        assert digest(target)==expected,'Canonical hash differs from its recorded identity'
        for row in request['pairs']:
            admission();source=validate(row)
            protected={str(p):metadata(p) for p in source.parent.rglob('*')
                       if p.is_file() and any(s in p.name for s in ('optimizer','scheduler','rng_state'))}
            status('checksumming_source',current=row['source'])
            sha=digest(source)
            assert sha==expected,'Source differs; preserve it and inspect'
            validate(row)
            receipt=dict(source=row['source'],target=request['target'],sha256=sha,
                         canonical_sha256=expected,source_metadata=row['metadata'],verified_at=time.time())
            durable(pending,receipt)
            temp=source.with_name('.'+source.name+'.share_'+str(os.getpid()))
            assert not os.path.lexists(temp)
            temp.symlink_to(os.path.relpath(target,source.parent))
            assert temp.resolve(strict=True)==target
            validate(row);os.replace(temp,source)
            assert source.is_symlink() and source.resolve(strict=True)==target
            assert metadata(target)==request['target_metadata']
            for name,record in protected.items():assert metadata(Path(name))==record
            receipt.update(replaced_at=time.time(),reclaimed_allocated_bytes=row['metadata']['allocated_bytes'])
            with ledger.open('a') as f:
                f.write(json.dumps(receipt)+'\n');f.flush();os.fsync(f.fileno())
            pending.unlink();completed+=1;reclaimed+=row['metadata']['allocated_bytes'];status('sharing')
        status('complete')
    except BaseException as error:
        status('needs_attention',error=repr(error),pending=pending.exists());raise


if __name__=='__main__':main(Path(sys.argv[1]))
