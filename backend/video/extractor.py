"""
Sample frames from a video at a configurable processing FPS.

Every extracted frame retains its exact video timestamp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    """A single frame with its source timestamp."""

    frame_index: int       # index within the *sampled* sequence (0-based)
    original_frame_num: int  # frame number in the original video
    timestamp: float       # seconds in the video
    path: str              # saved image path (empty until saved)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_frames(
    video_path: Union[str, Path],
    process_fps: Optional[float] = None,
    save_dir: Optional[Path] = None,
    save_frames: bool = True,
) -> List[ExtractedFrame]:
    """
    Sample frames from *video_path* at *process_fps*.

    Parameters
    ----------
    video_path : path to the source video.
    process_fps : frames per second to sample (default from settings).
    save_dir : directory to save extracted frame images.
    save_frames : if False, frames are only returned in memory
                  (path will be empty string).

    Returns
    -------
    List of ExtractedFrame objects, sorted by timestamp.
    """
    settings = get_settings()
    process_fps = process_fps or settings.process_fps
    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    try:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0:
            raise RuntimeError("Cannot determine video FPS.")

        # Interval between sampled frames (in terms of original frame numbers)
        frame_interval = max(1, round(original_fps / process_fps))

        # Prepare save directory
        if save_frames:
            if save_dir is None:
                video_stem = video_path.stem
                save_dir = settings.frames_dir / video_stem
            save_dir.mkdir(parents=True, exist_ok=True)

        frames: List[ExtractedFrame] = []
        sampled_idx = 0

        logger.info(
            "Extracting frames at %.1f FPS (interval=%d) from %s (%d total frames)",
            process_fps, frame_interval, video_path.name, total_frames,
        )

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                timestamp = round(frame_num / original_fps, 4)

                img_path = ""
                if save_frames and save_dir is not None:
                    img_name = f"frame_{sampled_idx:06d}_{timestamp:.3f}s.jpg"
                    img_path = str(save_dir / img_name)
                    cv2.imwrite(img_path, frame)

                ef = ExtractedFrame(
                    frame_index=sampled_idx,
                    original_frame_num=frame_num,
                    timestamp=timestamp,
                    path=img_path,
                )
                frames.append(ef)
                sampled_idx += 1

            frame_num += 1

        logger.info(
            "Extracted %d frames (%.1f FPS) from %s",
            len(frames), process_fps, video_path.name,
        )
        return frames
    finally:
        cap.release()


def load_frame_image(frame: ExtractedFrame) -> np.ndarray:
    """Load a saved frame image back into a numpy array (BGR)."""
    if not frame.path:
        raise ValueError("Frame was not saved to disk.")
    img = cv2.imread(frame.path)
    if img is None:
        raise RuntimeError(f"Cannot read frame image: {frame.path}")
    return img
