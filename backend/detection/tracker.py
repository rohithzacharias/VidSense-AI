"""
BoT-SORT object tracking via the Ultralytics tracking API.

Assigns persistent track IDs to YOLO detections across consecutive frames
so the system can reason about "the same person/object" over time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A single tracked detection in one frame."""

    timestamp: float
    frame_index: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameTracking:
    """All tracked objects for a single frame."""

    timestamp: float
    frame_index: int
    tracked_objects: List[TrackedObject] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "objects": [t.to_dict() for t in self.tracked_objects],
        }


class BoTSORTTracker:
    """
    Run BoT-SORT tracking through Ultralytics' model.track() API.

    Ultralytics internally handles the BoT-SORT tracker when we pass
    ``tracker='botsort.yaml'`` to model.track().  This avoids needing
    a separate BoT-SORT installation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
        tracker_config: Optional[str] = None,
    ):
        from ultralytics import YOLO

        settings = get_settings()
        self._model_name = model_name or settings.yolo_model
        self._confidence = confidence if confidence is not None else settings.yolo_confidence
        self._device = device or settings.yolo_device
        self._tracker_config = tracker_config or settings.tracker_type

        logger.info(
            "Loading YOLO model for tracking: %s  (tracker=%s, device=%s)",
            self._model_name, self._tracker_config, self._device,
        )
        self._model = YOLO(self._model_name)
        logger.info("Tracker ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_video_frames(
        self,
        frame_paths: List[str],
        timestamps: List[float],
        frame_indices: List[int],
    ) -> List[FrameTracking]:
        """
        Run BoT-SORT tracking on a sequence of saved frame images.

        The frames MUST be in chronological order for the tracker to
        maintain consistent IDs.

        Returns one FrameTracking per input frame.
        """
        all_tracking: List[FrameTracking] = []

        for path, ts, idx in zip(frame_paths, timestamps, frame_indices):
            img = cv2.imread(path)
            if img is None:
                logger.warning("Cannot read frame: %s — skipping", path)
                continue

            ft = self._track_single_frame(img, ts, idx)
            all_tracking.append(ft)

        unique_ids = set()
        for ft in all_tracking:
            for t in ft.tracked_objects:
                unique_ids.add(t.track_id)

        logger.info(
            "BoT-SORT: tracked %d frames → %d unique track IDs",
            len(all_tracking), len(unique_ids),
        )
        return all_tracking

    def track_video_file(
        self,
        video_path: Union[str, Path],
        process_fps: Optional[float] = None,
    ) -> List[FrameTracking]:
        """
        Run BoT-SORT tracking directly on a video file.

        This lets Ultralytics handle the video reading internally,
        which keeps the tracker state consistent across frames
        (important for track ID continuity).

        Parameters
        ----------
        video_path : path to the source video.
        process_fps : sample rate in FPS. Frames between samples are
                      still fed to the tracker to maintain ID continuity,
                      but only sampled frames are returned.
        """
        settings = get_settings()
        process_fps = process_fps or settings.process_fps
        video_path = Path(video_path).resolve()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if original_fps <= 0:
            raise RuntimeError("Cannot determine video FPS.")

        frame_interval = max(1, round(original_fps / process_fps))
        all_tracking: List[FrameTracking] = []

        logger.info(
            "Tracking video %s — %d total frames, sampling every %d",
            video_path.name, total_frames, frame_interval,
        )

        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0
        sampled_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Always run tracking to maintain ID continuity
                results = self._model.track(
                    source=frame,
                    conf=self._confidence,
                    device=self._device,
                    tracker=self._tracker_config,
                    persist=True,
                    verbose=False,
                )

                # Only record sampled frames
                if frame_num % frame_interval == 0:
                    ts = round(frame_num / original_fps, 4)
                    ft = self._parse_track_results(results, ts, sampled_idx)
                    all_tracking.append(ft)
                    sampled_idx += 1

                frame_num += 1
        finally:
            cap.release()

        unique_ids = set()
        for ft in all_tracking:
            for t in ft.tracked_objects:
                unique_ids.add(t.track_id)

        logger.info(
            "BoT-SORT complete: %d sampled frames, %d unique IDs",
            len(all_tracking), len(unique_ids),
        )
        return all_tracking

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _track_single_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> FrameTracking:
        """Track objects in a single frame image (must be called sequentially)."""
        results = self._model.track(
            source=image,
            conf=self._confidence,
            device=self._device,
            tracker=self._tracker_config,
            persist=True,
            verbose=False,
        )
        return self._parse_track_results(results, timestamp, frame_index)

    def _parse_track_results(
        self,
        results,
        timestamp: float,
        frame_index: int,
    ) -> FrameTracking:
        """Parse Ultralytics tracking results into our data model."""
        tracked: List[TrackedObject] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Track IDs are only present when tracking succeeds
            has_ids = boxes.id is not None

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self._model.names.get(cls_id, str(cls_id))
                conf = round(float(boxes.conf[i].item()), 4)
                bbox = [round(float(c), 1) for c in boxes.xyxy[i].tolist()]
                track_id = int(boxes.id[i].item()) if has_ids else -1

                tracked.append(TrackedObject(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox,
                ))

        return FrameTracking(
            timestamp=timestamp,
            frame_index=frame_index,
            tracked_objects=tracked,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_tracking(
        tracking: List[FrameTracking],
        output_path: Union[str, Path],
    ) -> Path:
        """Save tracking results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [ft.to_dict() for ft in tracking]
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Tracking data saved → %s", output_path)
        return output_path

    @staticmethod
    def summarize_tracks(
        tracking: List[FrameTracking],
    ) -> Dict[int, Dict]:
        """
        Build a summary of each unique track: class, first/last seen, frame count.

        Returns a dict keyed by track_id.
        """
        tracks: Dict[int, Dict] = {}

        for ft in tracking:
            for obj in ft.tracked_objects:
                tid = obj.track_id
                if tid not in tracks:
                    tracks[tid] = {
                        "track_id": tid,
                        "class_name": obj.class_name,
                        "first_seen": obj.timestamp,
                        "last_seen": obj.timestamp,
                        "frame_count": 0,
                    }
                tracks[tid]["last_seen"] = max(tracks[tid]["last_seen"], obj.timestamp)
                tracks[tid]["first_seen"] = min(tracks[tid]["first_seen"], obj.timestamp)
                tracks[tid]["frame_count"] += 1

        return tracks
