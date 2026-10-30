"""Wait for a free Bluesclues device, reserve it, and await a migration grant.

Only this job's reservations/affinity are changed. The grant is issued after the
Adroit copy is confirmed canceled without starting any reference attempts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path.home() / 'kevin'


def account_cpu_snapshot():
    """Read the kernel account counter, verifying that it covers every UID process."""
    group = '/user.slice/user-%d.slice' % os.getuid()
    complete = True
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if p.stat().st_uid == os.getuid():
                    membership = (p / 'cgroup').read_text().strip()
                    complete &= membership == '0::'+group or membership.startswith('0::'+group+'/')
            except OSError:
                if p.exists():
                    complete = False
    try:
        fields = dict(line.split() for line in
                      (Path('/sys/fs/cgroup') / group.lstrip('/') / 'cpu.stat').read_text().splitlines())
        return dict(usage_usec=int(fields['usage_usec']), time=time.monotonic(), complete=complete)
    except (OSError, ValueError, KeyError):
        return dict(usage_usec=None, time=time.monotonic(), complete=False)


def account_cpu_fraction(before, after, logical_cpus):
    elapsed = after['time']-before['time']
    if (not before['complete'] or not after['complete'] or elapsed <= 0 or not logical_cpus
            or before['usage_usec'] is None or after['usage_usec'] is None
            or after['usage_usec'] < before['usage_usec']):
        return None
    return (after['usage_usec']-before['usage_usec']) / 1e6 / elapsed / logical_cpus


def revalidated_capacity(previous, current):
    """A stale free-device result or incomplete CPU sample cannot authorize launch."""
    if (not previous.get('cpu_measurement_complete') or not current.get('cpu_measurement_complete')
            or previous.get('cpu_fraction') is None or current.get('cpu_fraction') is None):
        return None
    return dict(cpu_fraction=max(previous['cpu_fraction'], current['cpu_fraction']),
                cpu_measurement_complete=True,
                occupied=sorted(set(previous['occupied']) | set(current['occupied'])),
                free=sorted(set(previous['free']) & set(current['free'])))


def capacity():
    before = account_cpu_snapshot()
    time.sleep(2)
    after = account_cpu_snapshot()
    cpu = account_cpu_fraction(before, after, os.cpu_count())
    try:
        xml = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=10))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ET.ParseError) as error:
        # A failed observation is neither free capacity nor a failed experiment.
        # Let the caller's existing wait loop try again without reserving a GPU.
        return dict(cpu_fraction=None, cpu_measurement_complete=False,
                    cpu_measurement='unavailable', observed_at=time.time(),
                    occupied=[], free=[], gpu_observation_error=repr(error))
    occupied, free = set(), []
    for index, gpu in enumerate(xml.findall('gpu')):
        processes = gpu.findall('processes/process_info')
        for process in processes:
            try:
                if Path('/proc', process.findtext('pid')).stat().st_uid == os.getuid():
                    occupied.add(index)
            except OSError:
                # Unreadable ownership must not create apparent account headroom.
                occupied.add(index)
        used = int(gpu.findtext('fb_memory_usage/used').split()[0])
        if not processes and used < 128:
            free.append(index)
    return dict(cpu_fraction=cpu, cpu_measurement_complete=cpu is not None,
                cpu_measurement='kernel_user_slice' if cpu is not None else 'unavailable',
                observed_at=time.time(), occupied=sorted(occupied), free=free)


def reserve(snapshot):
    import fcntl
    from pick_original_base_collect import save
    if (not snapshot.get('cpu_measurement_complete') or snapshot['cpu_fraction'] is None
            or snapshot['cpu_fraction'] + 4 / os.cpu_count() > .20 or not snapshot['free']):
        return None
    directory = ROOT / 'dline'
    path = directory / 'gpu_slots.json'
    with (directory / 'gpu_slots.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(path.read_text()) if path.exists() else []
        live = [e for e in entries if Path('/proc', str(e['owner'])).exists()]
        # Other users do not use this account's registry. Re-read global GPU
        # occupancy and CPU capacity inside the lock immediately before reserving.
        snapshot = revalidated_capacity(snapshot, capacity())
        if snapshot is None:
            return None
        occupied = set(snapshot['occupied']) | {e['gpu'] for e in live}
        # Account for the graphics context Isaac may also create on physical GPU 0.
        candidates = [g for g in snapshot['free'] if g not in occupied and len(occupied | {g, 0}) <= 5]
        if snapshot['cpu_fraction'] + 4 / os.cpu_count() > .20 or not candidates:
            return None
        gpu = candidates[0]
        entries.append(dict(owner=os.getpid(), gpu=gpu, mb=45000))
        save(path, entries)
        return gpu


def release(gpu):
    import fcntl
    from pick_original_base_collect import save
    directory = ROOT / 'dline'
    path = directory / 'gpu_slots.json'
    with (directory / 'gpu_slots.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(path.read_text()) if path.exists() else []
        entries = [e for e in entries if not (e['owner'] == os.getpid() and e['gpu'] == gpu)]
        save(path, entries)


def main():
    import fcntl
    from pick_original_base_collect import save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    a = parser.parse_args()
    work = a.request.parent
    with (work / 'blue_queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (work / 'collection.started.json').exists()
        while True:
            if (work / 'adroit_won.json').exists():
                save(work / 'blue_queue.json', dict(time=time.time(), pid=os.getpid(), phase='adroit_won'))
                return
            snapshot = capacity()
            gpu = reserve(snapshot)
            if gpu is None:
                save(work / 'blue_queue.json', dict(time=time.time(), pid=os.getpid(), phase='waiting_capacity', capacity=snapshot))
                time.sleep(40)
                continue
            try:
                reservation = dict(time=time.time(), pid=os.getpid(), phase='reserved', gpu=gpu, expires=time.time()+120, capacity=snapshot)
                save(work / 'blue_queue.json', reservation)
                grant = work / 'blue_grant.json'
                while time.time() < reservation['expires']:
                    if (work / 'adroit_won.json').exists():
                        return
                    if grant.exists():
                        decision = json.loads(grant.read_text())
                        assert decision['adroit_cancelled'] is True and decision['adroit_attempts_started'] == 0
                        assert decision['reservation_pid'] == os.getpid() and decision['gpu'] == gpu
                        # Recheck graphics contexts and account CPU immediately before launch.
                        current = capacity()
                        assert gpu in current['free'] and len(set(current['occupied']) | {gpu, 0}) <= 5
                        assert current['cpu_fraction'] + 4 / os.cpu_count() <= .20
                        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
                        save(work / 'blue_queue.json', dict(reservation, time=time.time(), phase='running', cpu_mask=sorted(os.sched_getaffinity(0))))
                        with (work / 'supervisor.log').open('w') as log:
                            p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('pick_original_base_collect.py')),
                                                  str(a.request)], env=env, stdout=log, stderr=subprocess.STDOUT)
                            save(work / 'blue_worker.json', dict(pid=p.pid, gpu=gpu, started=time.time()))
                            code = p.wait()
                        (work / 'job.exit_code.txt').write_text(str(code)+'\n')
                        save(work / 'blue_queue.json', dict(reservation, time=time.time(), phase='finished' if code == 0 else 'failed', exit_code=code))
                        return
                    time.sleep(2)
            finally:
                release(gpu)
            save(work / 'blue_queue.json', dict(time=time.time(), pid=os.getpid(), phase='waiting_capacity', previous_reservation_expired=True))


if __name__ == '__main__':
    main()
