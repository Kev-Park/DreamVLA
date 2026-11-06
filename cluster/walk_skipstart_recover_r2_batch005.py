"""Recover the unfinished R2 batch without repeating its completed recordings.

One interrupted motion-53 attempt remains explicit in the recovery record.
This serial collector preserves the recipe and cannot start a fine-tune.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time


def completed_prefix(text, expected):
    starts = list(re.finditer(r'^\[INFO\] motion (\d+)_r0/\d+ \(written', text, re.M))
    completed = list(map(int, re.findall(r'\[rescue\] motion (\d+): pass A 4-rule', text)))
    assert 0 < len(completed) < len(expected)
    assert completed == expected[:len(completed)]
    assert [int(m.group(1)) for m in starts] == expected[:len(completed)+1]
    return text[:starts[len(completed)].start()], completed, expected[len(completed):]


def main():
    import h5py
    import gr00t_budget_chain as base
    from walk_skipstart_chain import FreshChain, TAG

    chain = FreshChain()
    chain.collection_gpu = 4
    # One replacement serial lane, not an additional parallel peer batch.
    chain.collection_reserve_bytes = 25_000_000_000
    name = 'r2_batch005'
    expected = list(range(50, 60))
    work = chain.work
    raw = chain.data/'raw'/name
    trained = base.read(work/f'{TAG}_r1.trained.json')
    checkpoint = Path(trained['checkpoint'])
    assert trained['steps'] == 67542 and base.checkpoint_ok(checkpoint)
    assert not (work/'abort.json').exists()
    assert not (work/(name+'.converted.json')).exists()
    old = Path('/proc/172639')
    if old.exists():
        fields = (old/'stat').read_text().rsplit(')', 1)[1].split()
        assert old.stat().st_uid == os.getuid()
        assert int(fields[19]) == 51971564 and fields[0] in ('Z', 'X')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc/'cmdline').read_bytes()
            state = (proc/'stat').read_text().rsplit(')', 1)[1].split()[0]
            assert not (b'collect_sonic_adapter.py' in cmd and TAG.encode() in cmd and state not in ('Z', 'X'))
        except FileNotFoundError:
            pass
    source_log = work/(name+'.log')
    original = source_log.read_text()
    prefix, completed, remaining = completed_prefix(original, expected)
    assert completed == [50, 51, 52] and remaining == list(range(53, 60))
    raw_ids = []
    for path in raw.rglob('*.hdf5'):
        with h5py.File(path, 'r', locking=False) as h:
            raw_ids.append(json.loads(h.attrs['metadata_json'])['motion_id'])
            assert len(h['data/demo_0/actions']) > 0
    assert sorted(raw_ids) == completed
    archive = work/(name+'.recovery_original')
    archive.mkdir(exist_ok=False)
    shutil.copy2(source_log, archive/source_log.name)
    receipt_path = work/(name+'.process.json')
    shutil.copy2(receipt_path, archive/receipt_path.name)
    receipt = base.read(receipt_path)
    assert receipt['pid'] == 172639
    recovery = dict(
        status='recollecting_unfinished_suffix', pid=os.getpid(), at=time.time(),
        original_log=str(archive/source_log.name),
        original_log_sha256=hashlib.sha256(original.encode()).hexdigest(),
        completed_ids_preserved=completed, remaining_ids=remaining,
        technical_aborted_attempts=[dict(motion_id=53, step_last_logged=400,
            actor_pid=172639, reason='Busy-loop stall; bounded two-core recovery test failed')],
        completed_student_attempts_target=10, seed=8,
        recipe_changed=False, random_stream_reinitialized_for_suffix=True,
        source_checkpoint=str(checkpoint), cpu_mask=sorted(os.sched_getaffinity(0)),
        gpu=4, original_gpu=0,
    )
    base.atomic(work/(name+'.recovery.json'), recovery)
    args = list(receipt['command'])
    # FreshChain.execute adds this exact fixed collector argument itself.
    pos = args.index('--skip-start-frames')
    assert args[pos+1] == '0'
    del args[pos:pos+2]
    args = [arg for arg in args if not str(arg).startswith('--kit_args=')]
    idsfile = work/(name+'.recovery.ids.txt')
    idsfile.write_text(' '.join(map(str, remaining))+'\n')
    args[args.index('--dagger-motion-ids')+1] = str(idsfile)
    assert args[args.index('--seed')+1] == '8'
    label = name+'_recovery'
    state_file = work/'dagger_peer_r2_gpu4.json'
    try:
        base.atomic(state_file, dict(pid=os.getpid(), round=2, batch=5,
            checkpoint=str(checkpoint), status='collecting', at=time.time()))
        gpu = chain.reserve(4, 20000)[0]
        try:
            chain.execute(args, label, gpu, chain.training_cwd)
        finally:
            chain.release(gpu)
        suffix = (work/(label+'.log')).read_text()
        base.verify_collection_logs([suffix], remaining)
        combined = ('[RECOVERY] Derived completed-attempt log; one aborted motion53 '
            'attempt is preserved in the immutable original log and recovery JSON.\n'
            + prefix + '\n' + suffix)
        base.verify_collection_logs([combined], expected)
        assert hashlib.sha256(source_log.read_bytes()).hexdigest() == recovery['original_log_sha256']
        tmp = source_log.with_suffix('.recovered.tmp')
        tmp.write_text(combined)
        tmp.replace(source_log)
        recovery.update(status='completed_suffix', suffix_log=str(work/(label+'.log')),
            completed_student_attempts=10, technical_aborted_attempt_count=1,
            completed_at=time.time())
        base.atomic(work/(name+'.recovery.json'), recovery)
        chain.finish_batch(name, expected, seed=8, checkpoint=checkpoint)
        for batch in range(6, 20):
            base.atomic(state_file, dict(pid=os.getpid(), round=2, batch=batch,
                checkpoint=str(checkpoint), status='collecting', at=time.time()))
            chain.batch(f'r2_batch{batch:03d}', list(range(batch*10, batch*10+10)),
                seed=8, checkpoint=checkpoint)
        base.atomic(state_file, dict(pid=os.getpid(), round=2, status='complete', at=time.time()))
    except BaseException as exc:
        base.atomic(state_file, dict(pid=os.getpid(), round=2, status='failed',
            error=repr(exc), at=time.time()))
        raise


if __name__ == '__main__':
    main()
