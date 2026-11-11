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


def verified_recovery_log(original_bytes, suffix, expected, recorded_digest):
    import gr00t_budget_chain as base
    # Progress bars contain carriage returns. read_text() normalizes them,
    # whereas read_bytes() preserves the immutable file's actual identity.
    text = original_bytes.decode().replace('\r\n', '\n').replace('\r', '\n')
    digest = hashlib.sha256(original_bytes).hexdigest()
    legacy_digest = hashlib.sha256(text.encode()).hexdigest()
    assert recorded_digest in (digest, legacy_digest)
    prefix, completed, remaining = completed_prefix(text, expected)
    assert completed == [50, 51, 52] and remaining == list(range(53, 60))
    base.verify_collection_logs([suffix], remaining)
    combined = ('[RECOVERY] Derived completed-attempt log; one aborted motion53 '
        'attempt is preserved in the immutable original log and recovery JSON.\n'
        + prefix + '\n' + suffix)
    base.verify_collection_logs([combined], expected)
    return combined, digest


def remaining_batches(chain, checkpoint):
    import gr00t_budget_chain as base
    state_file = chain.work/'dagger_peer_r2_gpu0.json'
    for batch in range(6, 20):
        base.atomic(state_file, dict(pid=os.getpid(), round=2, batch=batch,
            checkpoint=str(checkpoint), status='collecting', at=time.time()))
        chain.batch(f'r2_batch{batch:03d}', list(range(batch*10, batch*10+10)),
                    seed=8, checkpoint=checkpoint)
    base.atomic(state_file, dict(pid=os.getpid(), round=2, status='complete', at=time.time()))


def resume_completed_suffix(chain, name, expected, checkpoint):
    """Finalize an already closed suffix; never recollect completed attempts."""
    import gr00t_budget_chain as base
    work = chain.work
    recovery = base.read(work/(name+'.recovery.json'))
    assert recovery['status'] == 'recollecting_unfinished_suffix'
    assert recovery['source_checkpoint'] == str(checkpoint)
    assert recovery['completed_ids_preserved'] == [50, 51, 52]
    assert recovery['remaining_ids'] == list(range(53, 60))
    assert not Path('/proc', str(recovery['pid'])).exists()
    assert 'EXIT r2_batch005_recovery code=0' in (work/(name+'.recovery_controller.log')).read_text()
    source = work/(name+'.log')
    preserved = work/(name+'.recovery_original')/source.name
    assert Path(recovery['original_log']) == preserved
    original_bytes = source.read_bytes()
    assert original_bytes == preserved.read_bytes(), 'Original log content changed'
    suffix = (work/(name+'_recovery.log')).read_text()
    combined, digest = verified_recovery_log(original_bytes, suffix, expected,
                                              recovery['original_log_sha256'])
    recovery['legacy_normalized_text_sha256'] = recovery['original_log_sha256']
    recovery['original_log_sha256'] = digest
    recovery['log_hash_basis'] = 'raw_bytes'
    tmp = source.with_suffix('.recovered.tmp')
    tmp.write_text(combined)
    tmp.replace(source)
    recovery.update(status='completed_suffix', suffix_log=str(work/(name+'_recovery.log')),
        completed_student_attempts=10, technical_aborted_attempt_count=1,
        completed_at=time.time(), finalization_resume_pid=os.getpid(),
        finalization_only_no_new_collection=True)
    base.atomic(work/(name+'.recovery.json'), recovery)
    chain.finish_batch(name, expected, seed=8, checkpoint=checkpoint)
    remaining_batches(chain, checkpoint)


def main():
    import h5py
    import gr00t_budget_chain as base
    from walk_skipstart_chain import FreshChain, TAG

    chain = FreshChain()
    # Reuse the original physical device; do not add another account GPU.
    chain.collection_gpu = 0
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
        assert int(fields[19]) == 51971564 and fields[0] in ('T', 'Z', 'X')
        if fields[0] == 'T':
            assert int(fields[1]) == 47480
            parent = Path('/proc/47480')
            parent_fields = (parent/'stat').read_text().rsplit(')', 1)[1].split()
            assert int(parent_fields[19]) == 46427467
            assert b'walk_skipstart_chain.py' in (parent/'cmdline').read_bytes()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc/'cmdline').read_bytes()
            state = (proc/'stat').read_text().rsplit(')', 1)[1].split()[0]
            permitted_original = int(proc.name) == 172639 and state == 'T'
            assert not (b'collect_sonic_adapter.py' in cmd and TAG.encode() in cmd
                        and state not in ('Z', 'X') and not permitted_original)
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
    if (work/(name+'.recovery.json')).exists():
        assert sorted(raw_ids) == expected
        base.atomic(work/'dagger_peer_r2_gpu0.json', dict(pid=os.getpid(), round=2,
            batch=5, checkpoint=str(checkpoint), status='finalizing', at=time.time()))
        resume_completed_suffix(chain, name, expected, checkpoint)
        return
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
        original_log_sha256=hashlib.sha256(source_log.read_bytes()).hexdigest(),
        log_hash_basis='raw_bytes',
        completed_ids_preserved=completed, remaining_ids=remaining,
        technical_aborted_attempts=[dict(motion_id=53, step_last_logged=400,
            actor_pid=172639, reason='No logged progress after two-core test and five-minute four-core recovery; original preserved paused')],
        completed_student_attempts_target=10, seed=8,
        recipe_changed=False, random_stream_reinitialized_for_suffix=True,
        source_checkpoint=str(checkpoint), cpu_mask=sorted(os.sched_getaffinity(0)),
        gpu=0, original_gpu=0,
        original_actor_preserved_paused=old.exists(),
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
    state_file = work/'dagger_peer_r2_gpu0.json'
    try:
        base.atomic(state_file, dict(pid=os.getpid(), round=2, batch=5,
            checkpoint=str(checkpoint), status='collecting', at=time.time()))
        gpu = chain.reserve(0, 20000)[0]
        try:
            chain.execute(args, label, gpu, chain.training_cwd)
        finally:
            chain.release(gpu)
        suffix = (work/(label+'.log')).read_text()
        combined, digest = verified_recovery_log(source_log.read_bytes(), suffix, expected,
                                                   recovery['original_log_sha256'])
        assert digest == recovery['original_log_sha256']
        tmp = source_log.with_suffix('.recovered.tmp')
        tmp.write_text(combined)
        tmp.replace(source_log)
        recovery.update(status='completed_suffix', suffix_log=str(work/(label+'.log')),
            completed_student_attempts=10, technical_aborted_attempt_count=1,
            completed_at=time.time())
        base.atomic(work/(name+'.recovery.json'), recovery)
        chain.finish_batch(name, expected, seed=8, checkpoint=checkpoint)
        remaining_batches(chain, checkpoint)
    except BaseException as exc:
        base.atomic(state_file, dict(pid=os.getpid(), round=2, status='failed',
            error=repr(exc), at=time.time()))
        raise


if __name__ == '__main__':
    main()
