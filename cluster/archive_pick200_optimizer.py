"""Copy the two inactive pick200 optimizers to scoped Drive upload sessions.

This worker never deletes sources and never receives a general OAuth token.
Inference files, trainer-state JSON, the R2 optimizer and active R3 are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from upload_drive_session import atomic, transfer, validate_session

TAG = 'pick200_original_spares57_v2'
SOURCES = tuple(f'checkpoints/{TAG}_r{i}/checkpoint-{step}/optimizer.pt'
                for i, step in [(0, 35554), (1, 65953)])


def metadata(path):
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev, nlink=stat.st_nlink)


def checked_source(root, relative, expected=None):
    if relative not in SOURCES:
        raise ValueError('Optimizer is not an approved inactive candidate')
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Invalid source path')
    actual = metadata(path)
    if actual['nlink'] != 1 or (expected is not None and actual != expected):
        raise ValueError('Inventoried source changed')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['hash', 'upload'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    root = Path.home() / 'kevin'
    allowed = (root / 'dline/pick200_optimizer_archive_20261028').resolve()
    assert args.output.resolve().is_relative_to(allowed)
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {44})
    started = time.time()
    result_path = args.output / 'result.json'
    assert not result_path.exists(), 'Reconcile existing result rather than overwriting'
    try:
        if args.mode == 'hash':
            rows = []
            for relative in SOURCES:
                path = checked_source(root, relative)
                before = metadata(path)
                sha, md5 = hashlib.sha256(), hashlib.md5()
                count = 0
                previous = 0
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                        sha.update(chunk)
                        md5.update(chunk)
                        count += len(chunk)
                        if time.time() - previous > 2:
                            atomic(args.output / 'progress.json', dict(pid=os.getpid(),
                                   time=time.time(), phase='hashing', source=relative,
                                   read_bytes=count, total_bytes=before['bytes']))
                            previous = time.time()
                checked_source(root, relative, before)
                assert count == before['bytes']
                rows.append(dict(relative=relative, metadata=before,
                                 sha256=sha.hexdigest(), md5=md5.hexdigest()))
            result = dict(success=True, mode='hash', pid=os.getpid(), started=started,
                          completed=time.time(), sources=rows, source_retained=True)
        else:
            assert args.request and args.request.resolve().is_relative_to(allowed)
            assert not args.request.is_symlink()
            request = json.loads(args.request.read_text())
            args.request.unlink()  # Keep the scoped capability only in memory.
            validate_session(request['uri'])
            row = request['source']
            path = checked_source(root, row['relative'], row['metadata'])

            def progress(offset):
                atomic(args.output / 'progress.json', dict(pid=os.getpid(), time=time.time(),
                       started=started, acknowledged_bytes=offset, total_bytes=row['metadata']['bytes']))

            progress(0)
            with path.open('rb') as stream:
                response = transfer(stream, request['uri'], row['metadata']['bytes'],
                                    row['md5'], progress)
            checked_source(root, row['relative'], row['metadata'])
            if response.get('sha256Checksum') != row['sha256']:
                raise ValueError('Drive SHA256 does not match prepared source')
            result = dict(success=True, mode='upload', pid=os.getpid(), started=started,
                          completed=time.time(), file_id=response['id'],
                          source=row, bytes=int(response['size']), md5=response['md5Checksum'],
                          sha256=response['sha256Checksum'], source_retained=True)
    except BaseException as error:
        # Network exceptions may contain the capability. Never print their text.
        result = dict(success=False, mode=args.mode, pid=os.getpid(), started=started,
                      completed=time.time(), error_type=type(error).__name__, source_retained=True)
    atomic(result_path, result)
    print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
