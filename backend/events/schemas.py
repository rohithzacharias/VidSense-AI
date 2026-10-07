"""
Event data schemas.

The Event is the central contract of the entire VidSense system.
Later this same structure will be stored in PostgreSQL + pgvector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class Event:
    """A single semantic event detected in a video."""

    event_id: str                 # e.g. "evt_000001"
    video_id: str                 # e.g. "video_001"
    start_time: float             # seconds
    end_time: float               # seconds
    event_type: str               # e.g. "interaction", "movement", "idle", "alarm"
    description: str              # semantic description from VLM
    track_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0       # average detection confidence in the clip

    def to_dict(self) -> dict:
        return asdict(self)

    def format_time_range(self) -> str:
        """Human-readable time range like '00:10.0 – 00:15.0'."""
        return f"{_fmt(self.start_time)} – {_fmt(self.end_time)}"


@dataclass
class VideoEvents:
    """All events extracted from a single video."""

    video_id: str
    source_video: str
    total_duration: float
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "source_video": self.source_video,
            "total_duration": self.total_duration,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, output_path: Union[str, Path]) -> Path:
        """Save all events to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VideoEvents":
        """Load events from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        events = [Event(**e) for e in data["events"]]
        return cls(
            video_id=data["video_id"],
            source_video=data.get("source_video", ""),
            total_duration=data.get("total_duration", 0.0),
            events=events,
        )


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"
