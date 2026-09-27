"""Video writers used for recorded runs and replays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoWriter:
    """Frame sink writing either an mp4 (cv2) or a directory of PNGs."""

    path: Path
    fps: float = 20.0
    fourcc: str = "mp4v"
    _writer: object | None = None
    _frames: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.suffix.lower() not in {".mp4", ".avi", ".mkv"}:
            raise ValueError("unsupported video container %r" % self.path.suffix)

    def _ensure_open(self, frame_shape: tuple[int, int, int]) -> None:
        if self._writer is not None:
            return
        import cv2

        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_shape[0], frame_shape[1]
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("cv2 could not open a VideoWriter at %s" % self.path)
        self._writer = writer

    def append(self, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8 or rgb.ndim != 3:
            raise ValueError("expected (H, W, 3) uint8, got %s %s" % (rgb.dtype, rgb.shape))
        self._ensure_open(rgb.shape)
        import cv2

        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self._frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        return self._frames


def open_video_writer(path: str | Path, *, fps: float = 20.0) -> VideoWriter:
    return VideoWriter(path=Path(path), fps=fps)
