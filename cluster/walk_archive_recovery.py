"""Verify and repair interrupted raw archives without discarding old contents."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def checked_rows(rows):
    result = {}
    for row in rows:
        name = row['path']
        path = PurePosixPath(name)
        if (not name or '\\' in name or path.is_absolute() or
                '..' in path.parts or str(path) != name or name in result):
            raise ValueError('Unsafe or duplicate archive member: ' + repr(name))
        digest = row['sha256']
        if (not isinstance(row['bytes'], int) or row['bytes'] < 0 or
                len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Invalid archive fingerprint: ' + name)
        result[name] = row
    return result


def plan(expected, actual):
    wanted, present = checked_rows(expected), checked_rows(actual)
    extra = sorted(set(present) - set(wanted))
    if extra:
        raise ValueError('Unrecognized destination files; retain for review: ' + repr(extra))
    return [wanted[name] for name in sorted(wanted) if present.get(name) != wanted[name]]


def inventory(directory):
    root = Path(directory)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Archive root must be a real directory')
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in archive: ' + str(path))
        if not path.is_file():
            continue
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Archive changed during hashing: ' + str(path))
        rows.append({'path': path.relative_to(root).as_posix(),
                     'bytes': after.st_size, 'sha256': digest.hexdigest()})
    return rows


def promote(destination, staging, backup, expected, inspected):
    """Promote only verified repair files; preserve replaced files and a journal."""
    destination, staging, backup = map(Path, (destination, staging, backup))
    needed = plan(expected, inspected)
    expected = sorted(expected, key=lambda row: row['path'])
    roots = [p.resolve() for p in (destination, staging, backup)]
    for index, root in enumerate(roots):
        if any(root == other or root.is_relative_to(other) or other.is_relative_to(root)
               for other in roots[index + 1:]):
            raise ValueError('Recovery directories must be separate')
    if inventory(destination) != sorted(inspected, key=lambda row: row['path']):
        raise ValueError('Destination changed after inspection; no files replaced')
    if inventory(staging) != needed:
        raise ValueError('Staged repair differs from expected hashes; no files replaced')
    if backup.exists() or backup.is_symlink():
        raise ValueError('Backup already exists; reconcile its journal before proceeding')
    backup.mkdir(parents=True)
    destination.mkdir(parents=True, exist_ok=True)
    journal = {'phase': 'validated', 'expected': expected, 'completed': [], 'pending': None}

    def record():
        temporary = backup / 'PROMOTION.json.tmp'
        temporary.write_text(json.dumps(journal, indent=2) + '\n')
        temporary.replace(backup / 'PROMOTION.json')

    record()
    for row in needed:
        name = row['path']
        target, source, old = destination / name, staging / name, backup / name
        if any(p.is_symlink() for p in (target, source, old)):
            raise ValueError('Recovery member became a symlink: ' + name)
        journal.update(phase='promoting', pending=name)
        record()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, old)
            journal['phase'] = 'old_copy_preserved'
            record()
        os.replace(source, target)
        journal['completed'].append(name)
        journal.update(phase='promoting', pending=None)
        record()
    if inventory(destination) != expected:
        raise ValueError('Final archive failed full verification; inspect promotion journal')
    journal.update(phase='verified', files=len(expected))
    record()
    return journal
