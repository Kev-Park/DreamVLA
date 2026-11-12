import socket
import struct

import numpy as np
import pytest

from dreamvla_drawer_button.protocol import send_message, receive_message, message_text
from dreamvla_drawer_button.serve import (
    _build_observation,
    _normalize_action,
    STATE_DIMS,
    ACTION_DIMS,
)


def test_socket_roundtrip():
    left, right = socket.socketpair()
    try:
        send_message(left, dict(kind="infer", token=np.ones(64, np.float32)))
        msg = receive_message(right)
        assert message_text(msg, "kind") == "infer"
        np.testing.assert_array_equal(msg["token"], np.ones(64, np.float32))
    finally:
        left.close()
        right.close()


def test_object_arrays_rejected_before_transmission():
    with pytest.raises(ValueError):
        send_message(None, dict(bad=np.array([dict(a=1)], dtype=object)))


@pytest.mark.parametrize("size", [0, 64 * 1024 * 1024 + 1])
def test_oversized_frame(size):
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!Q", size))
        with pytest.raises(ValueError):
            receive_message(right)
    finally:
        left.close()
        right.close()


def test_premature_eof():
    left, right = socket.socketpair()
    left.close()
    try:
        with pytest.raises(EOFError):
            receive_message(right)
    finally:
        right.close()


@pytest.mark.parametrize("value", [np.array([1, 2]), np.array(123)])
def test_only_scalar_text(value):
    with pytest.raises(ValueError):
        message_text({"kind": value}, "kind")


def test_deployable_observation_contains_no_privileged_fields():
    msg = {name: np.zeros(width, np.float32) for name, width in STATE_DIMS.items()}
    msg.update(
        ego_view=np.zeros((480, 640, 3), np.uint8),
        prompt=np.array("open the drawer"),
        privileged=np.ones(200),
    )
    obs = _build_observation(msg)
    assert set(obs) == {"video", "state", "language"}
    assert sum(x.shape[-1] for x in obs["state"].values()) == 46
    msg["left_leg"][0] = np.nan
    with pytest.raises(ValueError):
        _build_observation(msg)


def test_only_40_step_full_action_chunks():
    act = {name: np.zeros((1, 40, width)) for name, width in ACTION_DIMS.items()}
    result = _normalize_action(act)
    assert sum(x.shape[-1] for x in result.values()) == 78
    act["motion_token"] = np.zeros((1, 20, 64))
    with pytest.raises(ValueError):
        _normalize_action(act)
