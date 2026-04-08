"""Utilities for writing robot_camera trajectories to HDF5
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def format_rollout_state(raw_state: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Placeholder for future rollout-state formatting.

    The caller is expected to provide downstream constraints before this
    hook is enabled.
    """

    raise NotImplementedError(
        "Rollout state formatting is intentionally left generic for now. "
        "Provide the downstream state layout before enabling this hook."
    )


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        child_group = group.create_group(key)
        for child_key, child_value in value.items():
            _write_value(child_group, child_key, child_value)
        return

    array_value = _to_numpy(value)
    if isinstance(array_value, str):
        group.create_dataset(key, data=np.array(array_value, dtype=h5py.string_dtype("utf-8")))
        return

    if isinstance(array_value, bytes):
        group.create_dataset(key, data=np.array(array_value, dtype=h5py.string_dtype("utf-8")))
        return

    if isinstance(array_value, np.ndarray) and array_value.dtype.kind in {"U", "O"}:
        group.create_dataset(key, data=np.array(array_value, dtype=h5py.string_dtype("utf-8")))
        return

    group.create_dataset(key, data=array_value, compression="gzip")


@dataclass
class RolloutRecorder:
    """Write a single rollout to a standalone HDF5 file."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_rollout(
        self,
        file_name: str,
        *,
        frames: np.ndarray,
        raw_state: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> Path:
        """Write rollout data to a new HDF5 file and return the file path."""

        if frames is None or frames.size == 0:
            raise ValueError("Images are mandatory: expected non-empty frame tensor for every rollout.")

        file_path = self.output_dir / file_name
        with h5py.File(file_path, "w") as handle:
            handle.attrs["metadata_json"] = json.dumps(_json_safe(metadata))
            handle.create_dataset("images", data=frames, compression="gzip", chunks=True)

            if raw_state is not None:
                state_group = handle.create_group("state")
                _write_value(state_group, "raw", raw_state)

        return file_path