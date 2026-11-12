"""Loopback-only, non-pickle client shared by Isaac and GR00T runtimes."""

import socket
import numpy as np

from .protocol import message_text, receive_message, send_message


class BridgeClient:
    def __init__(self, port, timeout=300):
        self.connection = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.policy_sha256 = message_text(self.call({"kind": "ping"}), "policy_sha256")

    def call(self, request):
        send_message(self.connection, request)
        result = receive_message(self.connection)
        if message_text(result, "kind") != "ok":
            raise RuntimeError(message_text(result, "message"))
        return result

    def reset(self, seed):
        self.call({"kind": "reset", "seed": np.asarray(seed, dtype=np.int64)})

    def close(self):
        self.connection.close()
