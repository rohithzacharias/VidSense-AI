"""
High-level VLM inference for temporal clips.

Given a TemporalClip and its tracking context, run Qwen2.5-VL and
return a semantic description of the event.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from backend.vlm.model import QwenVLModel
from backend.vlm.prompts import SYSTEM_PROMPT, build_clip_analysis_prompt
from backend.video.clips import TemporalClip
from backend.detection.tracker import FrameTracking

logger = logging.getLogger(__name__)


def _get_clip_track_summaries(
    clip: TemporalClip,
    tracking_data: List[FrameTracking],
) -> List[Dict]:
    """
    Extract unique tracked objects that appear within a clip's time window.

    Returns a de-duplicated list of {track_id, class_name}.
    """
    seen: Dict[int, str] = {}

    for ft in tracking_data:
        if ft.timestamp < clip.start_time or ft.timestamp >= clip.end_time:
            continue
        for obj in ft.tracked_objects:
            if obj.track_id >= 0 and obj.track_id not in seen:
                seen[obj.track_id] = obj.class_name

    return [
        {"track_id": tid, "class_name": cls}
        for tid, cls in sorted(seen.items())
    ]


def run_vlm_on_clip(
    model: QwenVLModel,
    clip: TemporalClip,
    tracking_data: List[FrameTracking],
    max_tokens: Optional[int] = None,
) -> str:
    """
    Run Qwen2.5-VL on a single temporal clip.

    Parameters
    ----------
    model : loaded QwenVLModel instance.
    clip : the temporal clip (must have vlm_frames populated).
    tracking_data : tracking results covering the clip's time range.
    max_tokens : override for generation length.

    Returns
    -------
    The VLM's semantic description of the clip.
    """
    frames = clip.vlm_frames
    if not frames:
        logger.warning("Clip %d has no VLM frames — skipping", clip.clip_index)
        return "No frames available for analysis."

    # Collect image paths
    image_paths = [f.path for f in frames if f.path]
    if not image_paths:
        logger.warning("Clip %d: no saved frame images — skipping", clip.clip_index)
        return "No frame images available for analysis."

    # Build tracking context
    track_summaries = _get_clip_track_summaries(clip, tracking_data)

    # Build prompt
    user_prompt = build_clip_analysis_prompt(
        start_time=clip.start_time,
        end_time=clip.end_time,
        track_summaries=track_summaries,
    )

    logger.info(
        "VLM → Clip %d [%.1fs–%.1fs] with %d frames, %d tracks",
        clip.clip_index, clip.start_time, clip.end_time,
        len(image_paths), len(track_summaries),
    )

    description = model.chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_paths=image_paths,
        max_tokens=max_tokens,
    )

    logger.info(
        "VLM ← Clip %d: %s",
        clip.clip_index,
        description[:120] + ("…" if len(description) > 120 else ""),
    )

    return description
