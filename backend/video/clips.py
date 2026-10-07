"""
Divide the sampled-frame timeline into fixed-duration temporal clips
and select representative frames for VLM input.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from backend.video.extractor import ExtractedFrame
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TemporalClip:
    """A fixed-duration window of the video with its representative frames."""

    clip_index: int
    start_time: float  # seconds
    end_time: float    # seconds
    all_frames: List[ExtractedFrame] = field(default_factory=list)
    vlm_frames: List[ExtractedFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def create_temporal_clips(
    frames: List[ExtractedFrame],
    video_duration: float,
    clip_duration: Optional[float] = None,
) -> List[TemporalClip]:
    """
    Divide the extracted frames into temporal windows of *clip_duration* seconds.

    Parameters
    ----------
    frames : sorted list of ExtractedFrame from the extractor.
    video_duration : total video duration in seconds.
    clip_duration : window size in seconds (default from settings).

    Returns
    -------
    List of TemporalClip objects, each containing the frames that fall
    within its time window.
    """
    settings = get_settings()
    clip_duration = clip_duration or settings.clip_duration

    num_clips = max(1, math.ceil(video_duration / clip_duration))
    clips: List[TemporalClip] = []

    for i in range(num_clips):
        start = round(i * clip_duration, 4)
        end = round(min((i + 1) * clip_duration, video_duration), 4)

        clip_frames = [f for f in frames if start <= f.timestamp < end]
        # Include frames exactly at end time for the last clip
        if i == num_clips - 1:
            clip_frames = [f for f in frames if start <= f.timestamp <= end]

        clip = TemporalClip(
            clip_index=i,
            start_time=start,
            end_time=end,
            all_frames=clip_frames,
        )
        clips.append(clip)

    logger.info(
        "Created %d temporal clips (%.1fs each) from %d frames",
        len(clips), clip_duration, len(frames),
    )
    return clips


def select_vlm_frames(
    clip: TemporalClip,
    max_frames: Optional[int] = None,
) -> List[ExtractedFrame]:
    """
    Select a subset of representative frames from a clip for VLM input.

    The selection is spread evenly across the temporal window to ensure
    good coverage rather than clustering frames at one point.

    Parameters
    ----------
    clip : the temporal clip to select from.
    max_frames : maximum number of frames to select (default from settings).

    Returns
    -------
    List of selected ExtractedFrame objects, ordered by timestamp.
    Also sets clip.vlm_frames as a side-effect.
    """
    settings = get_settings()
    max_frames = max_frames or settings.vlm_frames_per_clip

    available = clip.all_frames
    if not available:
        clip.vlm_frames = []
        return []

    n = len(available)
    if n <= max_frames:
        clip.vlm_frames = list(available)
        return clip.vlm_frames

    # Evenly spaced indices across the available frames
    indices = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    selected = [available[i] for i in unique_indices]
    clip.vlm_frames = selected

    logger.debug(
        "Clip %d: selected %d/%d frames for VLM (%.2fs–%.2fs)",
        clip.clip_index, len(selected), n, clip.start_time, clip.end_time,
    )
    return selected
