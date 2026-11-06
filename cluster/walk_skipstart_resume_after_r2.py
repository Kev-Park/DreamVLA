"""Resume the approved study only after its exact serial R2 owner completes.

This waits for a PID, verifies every collection/archive receipt, then replaces
itself with the normal chain. Fine-tunes remain routed to Adroit; this script
does not submit jobs or change account-wide resource policies.
"""
import os
from pathlib import Path
import select
import shutil
import sys
import time

import gr00t_budget_chain as base

# Keep the metadata gate independent of simulation-only module imports.
TAG = 'walk200_v3_skipstart_20261102'


def process_identity(pid):
    proc = Path('/proc')/str(pid)
    try:
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], start_ticks=int(fields[19]),
                    owned=proc.stat().st_uid == os.getuid(),
                    command=(proc/'cmdline').read_bytes())
    except FileNotFoundError:
        return None


def wait_owner(pid, start_ticks):
    identity = process_identity(pid)
    if identity is None or identity['state'] in ('Z', 'X'):
        return
    assert identity['owned'] and identity['start_ticks'] == start_ticks
    assert b'walk_skipstart_recover_r2_batch005.py' in identity['command']
    if hasattr(os, 'pidfd_open'):
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            return
        try:
            # Recheck identity after obtaining the handle to prevent PID reuse.
            identity = process_identity(pid)
            if identity is not None:
                assert identity['start_ticks'] == start_ticks
            select.select([fd], [], [])
        finally:
            os.close(fd)
    else:
        while True:
            identity = process_identity(pid)
            if identity is None or identity['state'] in ('Z', 'X'):
                return
            assert identity['start_ticks'] == start_ticks
            time.sleep(30)


def verified_collection(work):
    """Require the real complete dataset and matching raw archive for each batch."""
    reports = []
    for k in range(20):
        name = f'r2_batch{k:03d}'
        report = base.read(work/(name+'.converted.json'))
        assert report['batch'] == name
        assert report['requested_ids'] == list(range(k*10, (k+1)*10))
        row_ids = [row['motion_id'] for row in report['rows']]
        assert len(row_ids) == len(set(row_ids))
        assert set(row_ids) <= set(report['requested_ids'])
        assert report['retained'] == sum(row['passing'] for row in report['rows'])
        assert report['frames'] == sum(row['frames'] for row in report['rows'] if row['passing'])
        archive = base.read(work/(name+'.archived.json'))
        assert archive['batch'] == name and archive['checksum_verified'] is True
        assert archive['files'] == report['archive_files']
        if report['retained']:
            info = base.read(Path(report['dataset'])/'meta/info.json')
            assert info['total_episodes'] == report['retained']
            assert info['total_frames'] == report['frames']
        else:
            assert report['dataset'] is None and report['frames'] == 0
        reports.append(report)
    return dict(batches=reports, student_attempts=200)


def main():
    work = base.ROOT/'dline'/TAG
    config = base.read(work/'r2_deferred_main_config.json')
    state_path = work/'r2_deferred_main_state.json'
    try:
        grant = base.read(work/'four_core_collection_eval_user_approval.json')
        assert grant['authorized'] is True and grant['maximum_shared_cores'] == 4
        assert grant['scope'] == TAG+' collection and evaluation through round3 only'
        assert set(os.sched_getaffinity(0)) == set(grant['shared_mask'])
        base.atomic(state_path, dict(pid=os.getpid(), status='waiting_for_serial_owner',
            owner_pid=config['owner_pid'], owner_start_ticks=config['owner_start_ticks'],
            at=time.time()))
        wait_owner(config['owner_pid'], config['owner_start_ticks'])
        peer = base.read(work/'dagger_peer_r2_gpu0.json')
        assert peer['pid'] == config['owner_pid'] and peer['status'] == 'complete'
        assert not (work/'abort.json').exists()
        evaluation = base.read(work/f'{TAG}_r1.eval.json')
        assert evaluation['episodes'] == 100
        assert sorted(e['motion_id'] for e in evaluation['per_episode']) == list(range(100))
        collection = verified_collection(work)
        receipt = work/'collection_r2.json'
        if receipt.exists():
            assert base.read(receipt) == collection
        else:
            # The checkpoint archiver uses this aggregate completion receipt.
            # Publishing it here lets an eligible old checkpoint free space
            # before the next cumulative dataset merge.
            base.atomic(receipt, collection)
        while shutil.disk_usage(work).free < 25_000_000_000:
            assert not (work/'abort.json').exists()
            base.atomic(state_path, dict(pid=os.getpid(), status='waiting_for_merge_space',
                free_bytes=shutil.disk_usage(work).free, required_bytes=25_000_000_000,
                at=time.time()))
            time.sleep(30)
        routing = base.read(work/'routing.json')
        assert all(routing['finetune'][f'{TAG}_r{k}'] == 'adroit' for k in (2, 3))
        old_pid = int((work/'controller.lock').read_text())
        old = process_identity(old_pid)
        assert (old is None or old['state'] in ('Z', 'X')
                or b'walk_skipstart_chain.py' not in old['command']), 'Another main is active'
        for proc in Path('/proc').iterdir():
            if proc.name.isdigit():
                try:
                    identity = process_identity(int(proc.name))
                except PermissionError:
                    continue
                if identity and identity['owned'] and identity['state'] not in ('Z', 'X'):
                    command = identity['command']
                    assert not (b'collect_sonic_adapter.py' in command and TAG.encode() in command)
        assert not (work/'abort.json').exists()
        base.atomic(state_path, dict(pid=os.getpid(), status='exec_normal_chain',
            verified_student_attempts=200, at=time.time()))
        os.execv(sys.executable, [sys.executable, '-u', str(base.HERE/'walk_skipstart_chain.py')])
    except BaseException as exc:
        base.atomic(state_path, dict(pid=os.getpid(), status='stopped_for_review',
            error=repr(exc), at=time.time()))
        raise


if __name__ == '__main__':
    main()
