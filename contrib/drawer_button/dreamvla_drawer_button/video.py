"""Synchronous exporter-compatible RGB encoder with an explicit flush boundary.

Avoids depending on an uncommitted threaded-writer fix in the SONIC checkout.
Only installed in this converter process; no dependency files are modified.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np


class VideoWriter:
    def __init__(self, output_path, width, height, fps, codec="h264", buffer_size=50):
        import av

        self.output_path = str(output_path)
        self.width, self.height = width, height
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(self.output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=Fraction(str(fps)))
        self.stream.width = width
        self.stream.height = height
        self.closed = False

    def add_frame(self, rgb):
        import av

        if (
            self.closed
            or rgb.shape != (self.height, self.width, 3)
            or rgb.dtype != np.uint8
        ):
            raise ValueError(
                "Writer is closed or frame is not correctly sized RGB uint8"
            )
        for packet in self.stream.encode(
            av.VideoFrame.from_ndarray(rgb, format="rgb24")
        ):
            self.container.mux(packet)

    def stop(self):
        if not self.closed:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.closed = True
        return self.output_path

    def cancel(self):
        if not self.closed:
            self.container.close()
            self.closed = True
        Path(self.output_path).unlink(missing_ok=True)
