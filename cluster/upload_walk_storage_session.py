"""Upload one approved inactive walking artifact; never delete source data.

The coordinator retains OAuth. Workers receive only one scoped upload session.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from upload_adroit_archive_session import atomic, transfer, validate_session

PREFIXES = (
    'checkpoints/T20maxf_run01/',
    'checkpoints/T20maxf_dagger2_run01/',
    'checkpoints/walk200_v3_dagger57_20261013_r0/',
    'checkpoints/walk200_v3_dagger57_20261013_r3/',
    'checkpoints/walk200_v3_late1s_20261024_r3/checkpoint-109629/',
    'dline/walk_checkpoint_block_recovery_20261028/staged_checkpoint/',
    'dline/walk200_v3_late1s_20261024/raw/',
    'dline/walk200_v3_histwindow65_20261029/raw/',
)


def source(root, row):
    relative = row['relative']
    if not any(relative.startswith(prefix) for prefix in PREFIXES):
        raise ValueError('Source is outside approved walking artifacts')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Source link or path changed')
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino) != (
            row['bytes'], row['mtime_ns'], row['device'], row['inode']):
        raise ValueError('Source identity changed')
    return path


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
            md5.update(block)
    return {'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = (Path.home() / 'kevin').resolve()
    allowed = root / 'dline/walk_storage_offload_20261029'
    if not args.request.resolve().is_relative_to(allowed) or not args.output.resolve().is_relative_to(allowed):
        raise ValueError('Runtime must stay in the owned offload directory')
    args.output.mkdir(parents=True, exist_ok=True)
    result = args.output / 'result.json'
    if result.exists():
        raise ValueError('Reconcile the existing result before another worker')
    started = time.time()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    try:
        request = json.loads(args.request.read_text())
        args.request.unlink()
        validate_session(request['uri'])
        row = request['source']
        path = source(root, row)
        digest = hashes(path)
        if digest != {k: row[k] for k in ('sha256', 'md5')}:
            raise ValueError('Source hashes differ from inventory')
        source(root, row)
        atomic(args.output / 'source_integrity.json', dict(digest, bytes=row['bytes']))

        def progress(offset):
            atomic(args.output / 'progress.json', {'pid': os.getpid(), 'at': time.time(),
                   'acknowledged_bytes': offset, 'total_bytes': row['bytes']})

        def quota_wait(offset, attempt, delay):
            atomic(args.output / 'progress.json', {'pid': os.getpid(), 'at': time.time(),
                   'acknowledged_bytes': offset, 'total_bytes': row['bytes'],
                   'phase': 'drive_rate_limit_wait', 'retry': attempt,
                   'next_query_at': time.time() + delay})

        with path.open('rb') as stream:
            metadata = transfer(stream, request['uri'], row['bytes'], row['md5'], progress,
                                expected_sha256=row['sha256'], quota_wait=quota_wait)
        source(root, row)
        value = {'success': True, 'pid': os.getpid(), 'started': started, 'at': time.time(),
                 'source_retained': True, 'oauth_received': False, 'drive_metadata': metadata, **digest}
    except Exception as error:
        # Network exceptions can contain the scoped capability; never print them.
        value = {'success': False, 'pid': os.getpid(), 'at': time.time(),
                 'error_type': type(error).__name__, 'source_retained': True}
    atomic(result, value)
    print(json.dumps(value), flush=True)
    return 0 if value['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
