"""CUDA ordering for SONIC's Torch-owned, ONNX Runtime IO-bound buffers."""

from __future__ import annotations

import torch


class SonicCudaExecution:
    """Own a non-default stream and order both producers and consumers.

    Passing Torch's default stream pointer (0) leaves ORT's
    has_user_compute_stream disabled. Keep a real stream alive for the entire
    session lifetime and explicitly depend on the caller's buffer writes.
    """

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.stream = torch.cuda.Stream(device=device_id)

    def providers(self):
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": str(self.device_id),
                    "user_compute_stream": str(self.stream.cuda_stream),
                },
            ),
            "CPUExecutionProvider",
        ]

    def validate(self, session) -> None:
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "SONIC requested CUDA but ONNX Runtime did not enable it"
            )
        options = session.get_provider_options()["CUDAExecutionProvider"]
        if (
            options.get("has_user_compute_stream") != "1"
            or int(options.get("user_compute_stream", "0")) != self.stream.cuda_stream
        ):
            raise RuntimeError(
                f"SONIC CUDA stream was not enabled correctly: {options}"
            )

    def run(self, session, binding) -> None:
        caller = torch.cuda.current_stream(self.device_id)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            session.run_with_iobinding(binding)
        caller.wait_stream(self.stream)
