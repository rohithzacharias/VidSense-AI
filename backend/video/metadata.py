"""
Extract metadata from a video file using OpenCV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Structured container for video metadata."""

    path: str
    fps: float
    frame_count: int
    duration: float  # seconds
    width: int
    height: int
    codec: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_video_metadata(video_path: Union[str, Path]) -> VideoMetadata:
    """
    Open *video_path* with OpenCV and return structured metadata.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If OpenCV cannot open the video.
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Decode fourcc into a human-readable string
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).strip()

        duration = frame_count / fps if fps > 0 else 0.0

        meta = VideoMetadata(
            path=str(video_path),
            fps=round(fps, 2),
            frame_count=frame_count,
            duration=round(duration, 3),
            width=width,
            height=height,
            codec=codec,
        )

        logger.info(
            "Video metadata — %dx%d  %.2f FPS  %d frames  %.1f s  codec=%s",
            width, height, fps, frame_count, duration, codec,
        )
        return meta
    finally:
        cap.release()
