"""Small, crash-safe metadata utilities shared across process environments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


write = write_json
