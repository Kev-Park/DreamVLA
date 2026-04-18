"""Python wrapper around SONIC's kinematic planner ONNX (planner_sonic.onnx).

Input/output signature documented at
GR00T-WholeBodyControl/docs/source/references/planner_onnx.md.

The planner is stateless between calls — all context comes in through
``context_mujoco_qpos`` (4 frames of past qpos). Typical inference latency
per the paper (§3.3) is <5 ms on CPU and ~12 ms on Jetson Orin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Canonical planner input names, in the order declared by the ONNX model.
# Matches docs/source/references/planner_onnx.md.
_PLANNER_INPUTS = (
    "context_mujoco_qpos",
    "target_vel",
    "mode",
    "movement_direction",
    "facing_direction",
    "height",
)

_PLANNER_INPUT_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "context_mujoco_qpos": ((1, 4, 36), np.float32),
    "target_vel":          ((1,),       np.float32),
    "mode":                ((1,),       np.int64),
    "movement_direction":  ((1, 3),     np.float32),
    "facing_direction":    ((1, 3),     np.float32),
    "height":              ((1,),       np.float32),
}


@dataclass
class PlannerOutput:
    """Return value of ``PlannerWrapper.run``.

    - ``mujoco_qpos``: shape ``(1, num_pred_frames, 36)`` — already truncated
      to the valid prefix. Padding beyond ``num_pred_frames`` has been dropped.
    - ``num_pred_frames``: Python int, number of valid frames produced.
    """

    mujoco_qpos: np.ndarray
    num_pred_frames: int


class PlannerWrapper:
    """Minimal onnxruntime wrapper.

    Example:
        planner = PlannerWrapper("/path/to/planner_sonic.onnx")
        out = planner.run(
            context_mujoco_qpos=context,           # (1, 4, 36)
            target_vel=np.array([0.3], dtype=np.float32),
            mode=np.array([1], dtype=np.int64),    # SLOW_WALK
            movement_direction=vel_world[None],    # (1, 3)
            facing_direction=facing_world[None],   # (1, 3)
            height=np.array([0.78], dtype=np.float32),
        )
        # out.mujoco_qpos: (1, N, 36), out.num_pred_frames: N
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        path = Path(onnx_path)
        if not path.exists():
            raise FileNotFoundError(f"Planner ONNX not found at {path}")

        if providers is None:
            # Prefer CUDA when available; fall back to CPU.
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
            if not providers:
                providers = ort.get_available_providers()

        self.session = ort.InferenceSession(str(path), providers=providers)

        # Validate input signature against the ONNX model's declared inputs.
        model_input_names = {inp.name for inp in self.session.get_inputs()}
        missing = set(_PLANNER_INPUTS) - model_input_names
        extra = model_input_names - set(_PLANNER_INPUTS)
        if missing:
            raise RuntimeError(
                f"planner_sonic.onnx is missing expected inputs: {missing}. "
                f"Model declares: {sorted(model_input_names)}"
            )
        if extra:
            # Not fatal — newer planner revisions may add inputs. Warn.
            import warnings
            warnings.warn(
                f"planner_sonic.onnx declares extra inputs {extra} not passed by this "
                f"wrapper; they'll be left at their ONNX defaults or will error at run-time."
            )

        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(
        self,
        *,
        context_mujoco_qpos: np.ndarray,
        target_vel: np.ndarray,
        mode: np.ndarray,
        movement_direction: np.ndarray,
        facing_direction: np.ndarray,
        height: np.ndarray,
    ) -> PlannerOutput:
        """Run one planner inference. All inputs are shape-validated + dtype-cast."""
        feeds: dict[str, np.ndarray] = {
            "context_mujoco_qpos": _ensure(context_mujoco_qpos, *_PLANNER_INPUT_SPECS["context_mujoco_qpos"]),
            "target_vel":          _ensure(target_vel,          *_PLANNER_INPUT_SPECS["target_vel"]),
            "mode":                _ensure(mode,                *_PLANNER_INPUT_SPECS["mode"]),
            "movement_direction":  _ensure(movement_direction,  *_PLANNER_INPUT_SPECS["movement_direction"]),
            "facing_direction":    _ensure(facing_direction,    *_PLANNER_INPUT_SPECS["facing_direction"]),
            "height":              _ensure(height,              *_PLANNER_INPUT_SPECS["height"]),
        }
        outputs = self.session.run(self.output_names, feeds)
        named = dict(zip(self.output_names, outputs))
        qpos_padded = named.get("mujoco_qpos")
        num_frames_arr = named.get("num_pred_frames")
        if qpos_padded is None or num_frames_arr is None:
            raise RuntimeError(
                f"Unexpected planner output names: {list(named.keys())}; "
                f"expected 'mujoco_qpos' and 'num_pred_frames'."
            )
        num_frames = int(np.asarray(num_frames_arr).item())
        return PlannerOutput(
            mujoco_qpos=qpos_padded[:, :num_frames, :],
            num_pred_frames=num_frames,
        )


def _ensure(arr: Any, expected_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Cast to expected dtype and validate shape."""
    out = np.asarray(arr, dtype=dtype)
    if out.shape != expected_shape:
        raise ValueError(
            f"Planner input shape mismatch: got {out.shape}, expected {expected_shape}"
        )
    return out
