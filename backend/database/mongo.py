"""
MongoDB client and storage for VidSense AI events.

Stores structured event JSONs directly into MongoDB on localhost:27017.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ReplaceOne
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "vidsense_ai"


class MongoManager:
    """Manages MongoDB connection and event CRUD operations."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self._client: Optional[MongoClient] = None

    def get_client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
        return self._client

    @property
    def db(self):
        return self.get_client()[self.db_name]

    @property
    def events_collection(self):
        return self.db["events"]

    @property
    def videos_collection(self):
        return self.db["videos"]

    def is_connected(self) -> bool:
        """Check if MongoDB server is reachable."""
        try:
            self.get_client().admin.command("ping")
            return True
        except Exception as e:
            logger.warning("MongoDB ping failed: %s", e)
            return False

    def save_events_from_dict(self, data: Dict[str, Any]) -> int:
        """
        Store video summary and its events in MongoDB.
        Uses upsert on (video_id, event_id) so repeated calls update cleanly.
        """
        video_id = data.get("video_id", "default_video")
        source_video = data.get("source_video", "")
        total_duration = data.get("total_duration", 0.0)
        events = data.get("events", [])

        # Update video record
        self.videos_collection.update_one(
            {"video_id": video_id},
            {
                "$set": {
                    "video_id": video_id,
                    "source_video": source_video,
                    "total_duration": total_duration,
                    "event_count": len(events),
                }
            },
            upsert=True,
        )

        if not events:
            return 0

        # Bulk upsert events
        ops = []
        for evt in events:
            evt_copy = dict(evt)
            evt_copy["video_id"] = video_id
            event_id = evt_copy.get("event_id")
            ops.append(
                ReplaceOne(
                    {"video_id": video_id, "event_id": event_id},
                    evt_copy,
                    upsert=True,
                )
            )

        result = self.events_collection.bulk_write(ops)
        count = result.upserted_count + result.modified_count + result.inserted_count
        logger.info("Saved %d events for video '%s' into MongoDB", count, video_id)
        return len(events)

    def import_event_file(self, json_path: Path) -> int:
        """Read an event JSON file from disk and persist into MongoDB."""
        path = Path(json_path)
        if not path.is_file():
            logger.warning("Event file does not exist: %s", path)
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.save_events_from_dict(data)

    def sync_local_event_files(self) -> int:
        """Scan outputs/events and import all JSON files into MongoDB."""
        settings = get_settings()
        events_dir = settings.events_output_dir
        if not events_dir.is_dir():
            return 0

        total_imported = 0
        for p in events_dir.glob("*_events.json"):
            try:
                count = self.import_event_file(p)
                total_imported += count
            except Exception as e:
                logger.error("Failed importing %s to MongoDB: %s", p, e)
        return total_imported

    def get_video_events(self, video_id: str) -> List[Dict[str, Any]]:
        """Fetch all events for a video sorted by start_time."""
        cursor = self.events_collection.find(
            {"video_id": video_id},
            {"_id": 0},
        ).sort("start_time", 1)
        return list(cursor)

    def get_events_for_timerange(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Fetch events overlapping [start_time, end_time].
        """
        query: Dict[str, Any] = {"video_id": video_id}
        if start_time is not None and end_time is not None:
            # Overlap condition: event.start_time <= end_time AND event.end_time >= start_time
            query["$and"] = [
                {"start_time": {"$lte": float(end_time)}},
                {"end_time": {"$gte": float(start_time)}},
            ]
        elif start_time is not None:
            query["end_time"] = {"$gte": float(start_time)}
        elif end_time is not None:
            query["start_time"] = {"$lte": float(end_time)}

        cursor = self.events_collection.find(query, {"_id": 0}).sort("start_time", 1)
        return list(cursor)

    def list_videos(self) -> List[Dict[str, Any]]:
        """List all videos stored in the database."""
        cursor = self.videos_collection.find({}, {"_id": 0}).sort("video_id", 1)
        return list(cursor)


_manager_instance: Optional[MongoManager] = None


def get_mongo_manager() -> MongoManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MongoManager()
    return _manager_instance
