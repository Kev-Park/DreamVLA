"""Upload one inventoried archive file through a single-file Drive session.

OAuth stays on the coordinating Windows machine. This utility receives only an
already authorized session URI, never removes source data, and emits no URI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request


CHUNK = 32 * 1024**2


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def acknowledged(headers, total):
    value = headers.get('Range')
    if value is None:
        return 0
    match = re.fullmatch(r'bytes=0-(\d+)', value)
    if not match:
        raise ValueError('Invalid upload acknowledgement')
    offset = int(match[1]) + 1
    if not 0 < offset <= total:
        raise ValueError('Upload acknowledgement exceeds source size')
    return offset


def validate_session(uri):
    parsed = urllib.parse.urlparse(uri)
    query = urllib.parse.parse_qs(parsed.query)
    if (parsed.scheme != 'https' or parsed.netloc != 'www.googleapis.com'
            or parsed.path != '/upload/drive/v3/files'
            or query.get('uploadType') != ['resumable']
            or len(query.get('upload_id', [])) != 1):
        raise ValueError('Unexpected single-file upload endpoint')


def transfer(stream, uri, total, digest, progress, opener=urllib.request.urlopen,
             sleep=time.sleep, chunk=CHUNK, retries=4, expected_sha256=None,
             quota_retries=8, quota_wait=None):
    validate_session(uri)
    if total <= 0 or chunk <= 0 or chunk % (256 * 1024):
        raise ValueError('Invalid source or chunk size')
    offset = 0
    query = True
    failures = 0
    quota_failures = 0
    while True:
        if query:
            data = b''
            headers = {'Content-Range': f'bytes */{total}', 'Content-Length': '0'}
        else:
            stream.seek(offset)
            data = stream.read(min(chunk, total - offset))
            if not data:
                raise ValueError('Source ended before declared size')
            headers = {'Content-Type': 'application/octet-stream',
                       'Content-Length': str(len(data)),
                       'Content-Range': f'bytes {offset}-{offset+len(data)-1}/{total}'}
        request = urllib.request.Request(uri, data=data, method='PUT', headers=headers)
        try:
            with opener(request, timeout=120) as response:
                result = json.load(response)
            if (int(result['size']) != total or result['md5Checksum'] != digest
                    or not result.get('id')):
                raise ValueError('Completed upload checksum or size mismatch')
            if expected_sha256 is not None and result.get('sha256Checksum') != expected_sha256:
                raise ValueError('Completed upload SHA256 mismatch or missing provider checksum')
            progress(total)
            return result
        except urllib.error.HTTPError as error:
            if error.code == 403:
                try:
                    reasons={item.get('reason') for item in json.load(error).get('error',{}).get('errors',[])}
                except (ValueError,AttributeError):reasons=set()
                if reasons & {'rateLimitExceeded','userRateLimitExceeded'}:
                    if quota_failures >= quota_retries:
                        raise RuntimeError('Upload quota unavailable repeatedly; source retained') from None
                    delay=min(64,60+2**quota_failures)
                    quota_failures+=1
                    if quota_wait is not None:quota_wait(offset,quota_failures,delay)
                    sleep(delay)
                    query=True # Reconcile accepted bytes before any further content.
                    continue
            if error.code == 308:
                confirmed = acknowledged(error.headers, total)
                if confirmed < offset:
                    raise ValueError('Upload acknowledgement moved backwards') from None
                if not query and confirmed <= offset:
                    raise ValueError('Upload made no acknowledged progress') from None
                if confirmed > offset:
                    failures = 0
                    quota_failures = 0
                elif offset == total:
                    if failures >= retries:
                        raise RuntimeError('Final upload metadata unavailable; source retained') from None
                    sleep(min(16, 2**failures))
                    failures += 1
                offset = confirmed
                progress(offset)
                # Even a complete byte acknowledgement needs a final metadata
                # response. Query again instead of sending an empty data chunk.
                query = offset == total
                continue
            if error.code not in (500, 502, 503, 504):
                raise RuntimeError(f'Upload HTTP {error.code}; source retained') from None
        except (OSError, TimeoutError):
            pass
        # Never resend an unacknowledged chunk. Query the saved session first.
        if failures >= retries:
            raise RuntimeError('Upload observation failed repeatedly; source retained')
        sleep(min(16, 2**failures))
        failures += 1
        query = True


def source_path(root, row):
    base = root / 'cold_storage/pick342_historical_archive_20261020'
    path = base / row['relative']
    if (not path.resolve().is_relative_to(base.resolve()) or path.is_symlink()
            or path.suffix != '.hdf5'):
        raise ValueError('Invalid archive source')
    stat = path.stat()
    actual = stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino, stat.st_nlink
    expected = row['bytes'], row['mtime_ns'], row['device'], row['inode'], 1
    if actual != expected:
        raise ValueError('Inventoried source changed')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.home() / 'kevin'
    allowed = (root / 'dline/drive_archive_sessions').resolve()
    if not args.request.resolve().is_relative_to(allowed) or not args.output.resolve().is_relative_to(allowed):
        raise ValueError('Runtime files must stay in the archive session directory')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result_path = args.output / 'result.json'
    try:
        request = json.loads(args.request.read_text())
        args.request.unlink()  # The session capability remains only in memory.
        validate_session(request['uri'])
        row = request['source']
        path = source_path(root, row)
        os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024**2), b''):
                sha.update(block)
                md5.update(block)
        if md5.hexdigest() != request['md5']:
            raise ValueError('Source differs from its original checksum-verified archival ledger')
        source_path(root, row)
        atomic(args.output / 'source_integrity.json', dict(md5=md5.hexdigest(),
               sha256=sha.hexdigest(), bytes=row['bytes'], verified_at=time.time()))
        def progress(offset):
            atomic(args.output / 'progress.json', dict(pid=os.getpid(), time=time.time(),
                   started=started, acknowledged_bytes=offset, total_bytes=row['bytes']))
        def quota_wait(offset,attempt,delay):
            atomic(args.output / 'progress.json',dict(pid=os.getpid(),time=time.time(),started=started,
                acknowledged_bytes=offset,total_bytes=row['bytes'],phase='drive_rate_limit_wait',
                quota_retry=attempt,next_query_at=time.time()+delay))
        progress(0)
        with path.open('rb') as stream:
            response = transfer(stream, request['uri'], row['bytes'], request['md5'], progress,
                                expected_sha256=sha.hexdigest() if request.get('require_provider_sha256') else None,
                                quota_wait=quota_wait)
        source_path(root, row)
        receipt = dict(success=True, pid=os.getpid(), started=started, completed=time.time(),
                       file_id=response['id'], bytes=int(response['size']), md5=response['md5Checksum'],
                       sha256=sha.hexdigest(), source_retained=True, oauth_received=False)
        if request.get('require_provider_sha256'):
            receipt['drive_metadata'] = response
    except Exception as error:
        # Exceptions from networking can contain the capability URI: never log
        # their repr or traceback. The local controller reconciles the session.
        receipt = dict(success=False, pid=os.getpid(), started=started, completed=time.time(),
                       error_type=type(error).__name__, source_retained=True, oauth_received=False)
    atomic(result_path, receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
