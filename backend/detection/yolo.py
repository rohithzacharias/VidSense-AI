"""
YOLO object detection using the Ultralytics library.

Processes sampled frames and returns per-frame detection results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single YOLO detection on one frame."""

    timestamp: float
    frame_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameDetections:
    """All detections for a single frame."""

    timestamp: float
    frame_index: int
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "objects": [d.to_dict() for d in self.detections],
        }


class YOLODetector:
    """Wrapper around Ultralytics YOLO for frame-level object detection."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO

        settings = get_settings()
        self._model_name = model_name or settings.yolo_model
        self._confidence = confidence if confidence is not None else settings.yolo_confidence
        self._device = device or settings.yolo_device

        logger.info("Loading YOLO model: %s  (device=%s, conf=%.2f)",
                     self._model_name, self._device, self._confidence)
        self._model = YOLO(self._model_name)
        logger.info("YOLO model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> FrameDetections:
        """
        Run detection on a single frame image (BGR numpy array).

        Returns a FrameDetections object containing all detections.
        """
        results = self._model.predict(
            source=image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self._model.names.get(cls_id, str(cls_id))
                conf = round(float(boxes.conf[i].item()), 4)
                bbox = [round(float(c), 1) for c in boxes.xyxy[i].tolist()]

                detections.append(Detection(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox,
                ))

        return FrameDetections(
            timestamp=timestamp,
            frame_index=frame_index,
            detections=detections,
        )

    def detect_frames(
        self,
        frame_paths: List[str],
        timestamps: List[float],
        frame_indices: List[int],
    ) -> List[FrameDetections]:
        """
        Run detection on a batch of saved frame images.
        """
        all_detections: List[FrameDetections] = []

        for path, ts, idx in zip(frame_paths, timestamps, frame_indices):
            img = cv2.imread(path)
            if img is None:
                logger.warning("Cannot read frame image: %s — skipping", path)
                continue
            fd = self.detect_frame(img, ts, idx)
            all_detections.append(fd)
            if fd.detections:
                logger.debug(
                    "Frame %d (%.2fs): %d detections",
                    idx, ts, len(fd.detections),
                )

        logger.info("YOLO: processed %d frames, total detections = %d",
                     len(all_detections),
                     sum(len(fd.detections) for fd in all_detections))
        return all_detections

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_detections(
        detections: List[FrameDetections],
        output_path: Union[str, Path],
    ) -> Path:
        """Save detection results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [fd.to_dict() for fd in detections]
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Detections saved → %s", output_path)
        return output_path
