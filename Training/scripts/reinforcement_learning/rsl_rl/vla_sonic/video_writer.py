"""Shared RGB video writer for simulator playback entrypoints."""
from pathlib import Path


class RGBVideoWriter:
    def __init__(self, path: Path, fps: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        self.frames = 0
        self._writer = None

    @property
    def frame_count(self):
        return self.frames

    def write(self, frame_rgb):
        if self._writer is None:
            import imageio
            self._writer = imageio.get_writer(str(self.path), fps=self.fps,
                                               codec='libx264', quality=7,
                                               macro_block_size=None)
        self._writer.append_data(frame_rgb)
        self.frames += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if not self.frames or not self.path.is_file() or not self.path.stat().st_size:
            raise RuntimeError('Video writer produced no output: '+str(self.path))
