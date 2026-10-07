"""
Event Engine — converts VLM output + tracking context into structured Events.

This is the bridge between raw AI output and the database-ready Event schema.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from backend.events.schemas import Event
from backend.video.clips import TemporalClip
from backend.detection.tracker import FrameTracking

logger = logging.getLogger(__name__)

# Simple heuristics for event type classification
_EVENT_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "interaction": [
        "open", "close", "pick", "grab", "touch", "push", "pull",
        "hand", "press", "lift", "hold", "place", "put", "drop",
    ],
    "movement": [
        "walk", "run", "move", "approach", "enter", "exit", "leave",
        "cross", "pass", "step", "turn", "come", "go",
    ],
    "alarm": [
        "alarm", "alert", "warning", "siren", "flash", "emergency",
    ],
    "communication": [
        "talk", "speak", "point", "gesture", "wave", "signal", "nod",
    ],
    "vehicle": [
        "drive", "park", "arrive", "depart", "truck", "car", "vehicle",
    ],
    "idle": [
        "stand", "sit", "wait", "still", "stationary", "no significant",
        "nothing", "no activity", "no notable",
    ],
}


class EventEngine:
    """Convert VLM descriptions + clip metadata into structured Events."""

    def __init__(self, video_id: str):
        self._video_id = video_id
        self._event_counter = 0

    def create_event(
        self,
        clip: TemporalClip,
        vlm_description: str,
        tracking_data: List[FrameTracking],
    ) -> Event:
        """
        Build a structured Event from a clip's VLM output.

        Parameters
        ----------
        clip : the temporal clip that was analysed.
        vlm_description : the VLM's semantic output for this clip.
        tracking_data : tracking results covering this clip's range.

        Returns
        -------
        A fully populated Event.
        """
        self._event_counter += 1
        event_id = f"evt_{self._event_counter:06d}"

        # Determine track IDs present in this clip
        track_ids = self._extract_track_ids(clip, tracking_data, vlm_description)

        # Compute average confidence from tracking data
        confidence = self._compute_confidence(clip, tracking_data)

        # Classify event type
        event_type = self._classify_event_type(vlm_description)

        event = Event(
            event_id=event_id,
            video_id=self._video_id,
            start_time=clip.start_time,
            end_time=clip.end_time,
            event_type=event_type,
            description=vlm_description,
            track_ids=sorted(track_ids),
            confidence=round(confidence, 4),
        )

        logger.info(
            "[EVENT] %s  %s  type=%s  tracks=%s",
            event_id, event.format_time_range(), event_type, track_ids,
        )
        logger.info("  → %s", vlm_description[:150])

        return event

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_track_ids(
        self,
        clip: TemporalClip,
        tracking_data: List[FrameTracking],
        vlm_description: str,
    ) -> List[int]:
        """
        Gather track IDs from two sources:
          1. Tracking data in the clip's time window.
          2. Track IDs mentioned in the VLM description (e.g. "Person #3").
        """
        ids: Set[int] = set()

        # From tracking data
        for ft in tracking_data:
            if ft.timestamp < clip.start_time or ft.timestamp >= clip.end_time:
                continue
            for obj in ft.tracked_objects:
                if obj.track_id >= 0:
                    ids.add(obj.track_id)

        # From VLM text (pattern: #<number>)
        mentioned = re.findall(r"#(\d+)", vlm_description)
        for m in mentioned:
            ids.add(int(m))

        return sorted(ids)

    @staticmethod
    def _compute_confidence(
        clip: TemporalClip,
        tracking_data: List[FrameTracking],
    ) -> float:
        """Average detection confidence for objects in the clip window."""
        confidences: List[float] = []
        for ft in tracking_data:
            if ft.timestamp < clip.start_time or ft.timestamp >= clip.end_time:
                continue
            for obj in ft.tracked_objects:
                confidences.append(obj.confidence)

        return sum(confidences) / len(confidences) if confidences else 0.0

    @staticmethod
    def _classify_event_type(description: str) -> str:
        """Simple keyword-based event type classification."""
        desc_lower = description.lower()

        best_type = "observation"
        best_score = 0

        for etype, keywords in _EVENT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > best_score:
                best_score = score
                best_type = etype

        return best_type
