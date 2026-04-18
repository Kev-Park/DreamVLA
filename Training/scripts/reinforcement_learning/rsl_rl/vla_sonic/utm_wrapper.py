"""Python wrapper around SONIC's UniversalTokenModule ONNX exports.

Two ONNX models together form the UTM pipeline:
- ``model_encoder.onnx``: tokenizer_obs → 64-D FSQ-quantized token.
- ``model_decoder.onnx``: token + proprioception history → joint targets.

Introspection-first design: on init, we read the input/output signature from
each ONNX model and expose it via ``describe()``. That way you can confirm
the downloaded weights' actual signature matches what the adapters expect
(rather than trusting documentation that lagged the release).

Expected encoder inputs for the "teleop" mode (see
GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml):

    encoder_mode_4                                        (4,)   float32
    motion_joint_positions_lowerbody_10frame_step5        (120,) float32
    motion_joint_velocities_lowerbody_10frame_step5       (120,) float32
    vr_3point_local_target                                (9,)   float32
    vr_3point_local_orn_target                            (12,)  float32  [3x4 quat]
    motion_anchor_orientation                             (6,)   float32  [rot6d]

Expected decoder inputs (exact names confirmed at load time via ``describe()``):

    token_state                                           (64,)  float32
    his_base_angular_velocity_10frame_step1               (12,)  float32
    his_body_joint_positions_10frame_step1                (116,) float32
    his_body_joint_velocities_10frame_step1               (116,) float32
    his_last_actions_10frame_step1                        (116,) float32
    his_gravity_dir_10frame_step1                         (12,)  float32

Decoder output is joint targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _OnnxIoSpec:
    """Introspected signature of an ONNX input or output tensor."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str

    def __str__(self) -> str:
        return f"{self.name}: {self.shape} {self.dtype}"


class UtmWrapper:
    """Two ONNX sessions + introspected signatures.

    Example:
        utm = UtmWrapper(
            encoder_onnx_path="/path/to/model_encoder.onnx",
            decoder_onnx_path="/path/to/model_decoder.onnx",
        )
        print(utm.describe())

        tokens = utm.run_encoder({
            "encoder_mode_4": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 4),
            # ... rest of teleop encoder inputs
        })

        actions = utm.run_decoder({
            "token_state": tokens,                       # (1, 64)
            # ... proprio history tensors
        })
    """

    def __init__(
        self,
        encoder_onnx_path: str | Path,
        decoder_onnx_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        enc_path = Path(encoder_onnx_path)
        dec_path = Path(decoder_onnx_path)
        if not enc_path.exists():
            raise FileNotFoundError(f"Encoder ONNX not found at {enc_path}")
        if not dec_path.exists():
            raise FileNotFoundError(f"Decoder ONNX not found at {dec_path}")

        if providers is None:
            available = ort.get_available_providers()
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ] or ort.get_available_providers()

        self.encoder = ort.InferenceSession(str(enc_path), providers=providers)
        self.decoder = ort.InferenceSession(str(dec_path), providers=providers)

        # Cache signatures for fast lookup + pretty-printing.
        self.encoder_inputs = tuple(_spec(io) for io in self.encoder.get_inputs())
        self.encoder_outputs = tuple(_spec(io) for io in self.encoder.get_outputs())
        self.decoder_inputs = tuple(_spec(io) for io in self.decoder.get_inputs())
        self.decoder_outputs = tuple(_spec(io) for io in self.decoder.get_outputs())

        self._encoder_output_names = [o.name for o in self.encoder_outputs]
        self._decoder_output_names = [o.name for o in self.decoder_outputs]

    # ---------- introspection ----------

    def describe(self) -> str:
        """Return a human-readable summary of encoder + decoder signatures."""
        lines: list[str] = []
        lines.append("=== UTM Encoder ===")
        lines.append("  Inputs:")
        for spec in self.encoder_inputs:
            lines.append(f"    {spec}")
        lines.append("  Outputs:")
        for spec in self.encoder_outputs:
            lines.append(f"    {spec}")
        lines.append("=== UTM Decoder ===")
        lines.append("  Inputs:")
        for spec in self.decoder_inputs:
            lines.append(f"    {spec}")
        lines.append("  Outputs:")
        for spec in self.decoder_outputs:
            lines.append(f"    {spec}")
        return "\n".join(lines)

    def input_names(self, which: str) -> list[str]:
        """Return ordered input names for ``which`` in {'encoder', 'decoder'}."""
        specs = {"encoder": self.encoder_inputs, "decoder": self.decoder_inputs}[which]
        return [s.name for s in specs]

    def dummy_feeds(self, which: str) -> dict[str, np.ndarray]:
        """Build a dict of zero-filled tensors matching a model's input signature.

        Dynamic dims (strings like 'batch_size') are instantiated as 1.
        Useful for smoke-testing whether a model can run at all.
        """
        specs = {"encoder": self.encoder_inputs, "decoder": self.decoder_inputs}[which]
        feeds: dict[str, np.ndarray] = {}
        for s in specs:
            concrete = tuple(1 if isinstance(d, str) or d is None else d for d in s.shape)
            dtype = _np_dtype(s.dtype)
            feeds[s.name] = np.zeros(concrete, dtype=dtype)
        return feeds

    # ---------- inference ----------

    def run_encoder(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Run the tokenizer encoder. Returns the token output (usually (1, 64))."""
        _validate_feeds(feeds, self.encoder_inputs)
        outputs = self.encoder.run(self._encoder_output_names, feeds)
        # Most UTM encoders have a single output (token/latent).
        if len(outputs) == 1:
            return outputs[0]
        # If multiple, return the full dict so caller can pick.
        return dict(zip(self._encoder_output_names, outputs))  # type: ignore[return-value]

    def run_decoder(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Run the control decoder. Returns the action output."""
        _validate_feeds(feeds, self.decoder_inputs)
        outputs = self.decoder.run(self._decoder_output_names, feeds)
        if len(outputs) == 1:
            return outputs[0]
        return dict(zip(self._decoder_output_names, outputs))  # type: ignore[return-value]


def _spec(io: Any) -> _OnnxIoSpec:
    return _OnnxIoSpec(
        name=io.name,
        shape=tuple(io.shape),
        dtype=_prettify_dtype(io.type),
    )


def _prettify_dtype(ort_type: str) -> str:
    """Turn 'tensor(float)' into 'float32', etc."""
    mapping = {
        "tensor(float)":  "float32",
        "tensor(double)": "float64",
        "tensor(int32)":  "int32",
        "tensor(int64)":  "int64",
        "tensor(bool)":   "bool",
        "tensor(uint8)":  "uint8",
    }
    return mapping.get(ort_type, ort_type)


def _np_dtype(pretty: str) -> np.dtype:
    return {
        "float32": np.float32,
        "float64": np.float64,
        "int32":   np.int32,
        "int64":   np.int64,
        "bool":    np.bool_,
        "uint8":   np.uint8,
    }.get(pretty, np.float32)


def _validate_feeds(feeds: dict[str, np.ndarray], specs: tuple[_OnnxIoSpec, ...]) -> None:
    """Fail fast if any required input is missing."""
    expected = {s.name for s in specs}
    got = set(feeds.keys())
    missing = expected - got
    if missing:
        raise ValueError(
            f"Missing ONNX inputs: {sorted(missing)}. "
            f"Provided: {sorted(got)}. Expected: {sorted(expected)}."
        )
