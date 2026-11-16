#!/usr/bin/env python3
"""Small, dependency-free protocol for local GR00T/Isaac closed-loop tests.

The two applications cannot share a Python environment.  Messages are framed
NPZ payloads so NumPy arrays cross the process boundary without pickle.  The
server is intentionally bound to loopback by its launcher.
"""

from __future__ import annotations

from io import BytesIO
import socket
import struct
import zipfile
from typing import Mapping

import numpy as np


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Peer closed the GR00T bridge connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection: socket.socket, fields: Mapping[str, object]) -> None:
    """Send one mapping of concrete NumPy-compatible values."""
    payload = BytesIO()
    arrays = {name: np.asarray(value) for name, value in fields.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays are forbidden by the bridge protocol")
    np.savez(payload, **arrays)
    raw = payload.getvalue()
    if len(raw) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"GR00T bridge message is too large: {len(raw)} bytes")
    connection.sendall(_HEADER.pack(len(raw)))
    connection.sendall(raw)


def receive_message(connection: socket.socket) -> dict[str, np.ndarray]:
    """Receive one mapping, refusing object arrays and oversized payloads."""
    (size,) = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
    if size <= 0 or size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid GR00T bridge message size: {size}")
    raw = _recv_exact(connection, size)
    with zipfile.ZipFile(BytesIO(raw)) as zipped:
        if sum(item.file_size for item in zipped.infolist()) > _MAX_MESSAGE_BYTES:
            raise ValueError("Expanded bridge payload exceeds the memory limit")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def message_text(message: Mapping[str, np.ndarray], key: str) -> str:
    """Read a scalar Unicode field from a decoded message."""
    if key not in message:
        raise KeyError(f"Missing protocol field: {key}")
    value = np.asarray(message[key])
    if value.size != 1 or value.dtype.kind not in ("U", "S"):
        raise ValueError(f"Protocol field {key!r} must be scalar, got {value.shape}")
    item = value.reshape(-1)[0]
    return item.decode("utf-8") if value.dtype.kind == "S" else str(item)
