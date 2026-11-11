"""Upload one final walking checkpoint shard through a scoped Drive session.

The source checkpoint is immutable and retained. OAuth remains local to the
coordinator; the worker receives only a single-file upload capability.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from upload_adroit_archive_session import atomic, transfer, validate_session

PREFIX = 'checkpoints/walk200_v3_late1s_20261024_r3/checkpoint-109629/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = (Path.home() / 'kevin').resolve()
    allowed = root / 'dline/walk_late_checkpoint_drive_relay_20261028'
    assert args.request.resolve().is_relative_to(allowed)
    assert args.output.resolve().is_relative_to(allowed)
    args.output.mkdir(parents=True, exist_ok=True)
    result = args.output / 'result.json'
    assert not result.exists(), 'Reconcile the existing upload result before any restart'
    started = time.time()
    try:
        request = json.loads(args.request.read_text())
        args.request.unlink()
        validate_session(request['uri'])
        row = request['source']
        assert row['relative'].startswith(PREFIX)
        path = root / row['relative']
        assert path.resolve().is_relative_to(root / PREFIX)
        assert not path.is_symlink() and path.suffix == '.safetensors'
        before = path.stat()
        assert before.st_size == row['bytes']
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpus[request['slot'] % len(cpus)]})
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                sha.update(block)
                md5.update(block)
        assert sha.hexdigest() == row['sha256'], 'Source differs from the published inventory'

        def progress(offset):
            atomic(args.output / 'progress.json', {'pid': os.getpid(), 'at': time.time(),
                   'acknowledged_bytes': offset, 'total_bytes': row['bytes']})

        def quota_wait(offset, attempt, delay):
            atomic(args.output / 'progress.json', {'pid': os.getpid(), 'at': time.time(),
                   'acknowledged_bytes': offset, 'total_bytes': row['bytes'],
                   'phase': 'rate_limit_wait', 'retry': attempt, 'next_query_at': time.time() + delay})

        with path.open('rb') as source:
            metadata = transfer(source, request['uri'], row['bytes'], md5.hexdigest(),
                                progress, expected_sha256=row['sha256'], quota_wait=quota_wait)
        after = path.stat()
        assert (before.st_size, before.st_ino, before.st_mtime_ns) == (
            after.st_size, after.st_ino, after.st_mtime_ns)
        value = {'success': True, 'at': time.time(), 'started': started,
                 'source_retained': True, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest(),
                 'bytes': row['bytes'], 'drive_metadata': metadata, 'oauth_received': False}
    except Exception as error:
        # Networking errors may contain the scoped capability. Never log their
        # text or traceback; preserve the session for status reconciliation.
        value = {'success': False, 'at': time.time(), 'started': started,
                 'error_type': type(error).__name__, 'source_retained': True}
    atomic(result, value)
    print(json.dumps(value), flush=True)
    return 0 if value['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
