"""
VidSense AI — FastAPI Server.

Serves REST endpoints for:
- Video upload & processing
- Event retrieval from MongoDB
- Question answering via Groq LLM with temporal window filtering
- Static video streaming & evidence frame serving
- Frontend UI serving
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.database.mongo import get_mongo_manager
from backend.llm.groq_service import get_groq_service
from backend.pipeline.process_video import process_video

logger = logging.getLogger("vidsense.api")
logging.basicConfig(level=logging.INFO)

settings = get_settings()
settings.ensure_dirs()

app = FastAPI(
    title="VidSense AI API",
    description="Video Understanding & Temporal Reasoning API",
    version="0.2.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class AskRequest(BaseModel):
    video_id: str = Field(..., description="Target video identifier")
    query: str = Field(..., description="User's question about the video")
    current_time: Optional[float] = Field(None, description="Current playhead time in seconds")
    time_range_start: Optional[float] = Field(None, description="Start of focused time window")
    time_range_end: Optional[float] = Field(None, description="End of focused time window")


class ProcessingStatusResponse(BaseModel):
    video_id: str
    status: str
    message: str


# Helper: find evidence frames for a time window
def _find_evidence_frames(video_id: str, start_time: float, end_time: float, max_frames: int = 4) -> List[Dict[str, Any]]:
    frames_dir = settings.frames_dir / video_id
    if not frames_dir.is_dir():
        return []

    # Frame naming: frame_000123_0012.30.jpg -> index, timestamp
    candidates = []
    for f in sorted(frames_dir.glob("*.jpg")):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            try:
                ts = float(parts[2])
                if start_time - 0.5 <= ts <= end_time + 0.5:
                    candidates.append({
                        "timestamp": round(ts, 2),
                        "time_formatted": f"{int(ts//60):02d}:{int(ts%60):02d}",
                        "url": f"/data/frames/{video_id}/{f.name}",
                        "filename": f.name,
                    })
            except ValueError:
                continue

    if not candidates:
        return []

    # Pick evenly spaced frames
    if len(candidates) <= max_frames:
        return candidates
    step = len(candidates) / max_frames
    return [candidates[int(i * step)] for i in range(max_frames)]


# Startup: sync local event JSONs into MongoDB
@app.on_event("startup")
async def on_startup():
    logger.info("Initializing VidSense AI backend …")
    mongo = get_mongo_manager()
    if mongo.is_connected():
        count = mongo.sync_local_event_files()
        logger.info("Connected to MongoDB. Synced %d events from outputs/events.", count)
    else:
        logger.warning("MongoDB connection could not be established at %s", mongo.uri)


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    mongo = get_mongo_manager()
    return {
        "status": "healthy",
        "mongodb_connected": mongo.is_connected(),
        "uploads_dir": str(settings.videos_upload_dir),
    }


@app.get("/api/videos")
def list_videos():
    """List all available uploaded and processed videos."""
    mongo = get_mongo_manager()
    db_videos = {v["video_id"]: v for v in mongo.list_videos()}

    # Also list physical files in uploads
    result = []
    found_stems = set()
    for vid_file in sorted(settings.videos_upload_dir.glob("*.*")):
        if vid_file.suffix.lower() not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            continue
        vid_id = vid_file.stem
        found_stems.add(vid_id)
        db_info = db_videos.get(vid_id, {})
        has_events = bool(db_info.get("event_count", 0))

        # Check for first frame thumbnail
        frames_dir = settings.frames_dir / vid_id
        thumbnail_url = None
        if frames_dir.is_dir():
            first_frame = next(frames_dir.glob("*.jpg"), None)
            if first_frame:
                thumbnail_url = f"/data/frames/{vid_id}/{first_frame.name}"

        result.append({
            "video_id": vid_id,
            "filename": vid_file.name,
            "url": f"/data/videos/uploads/{vid_file.name}",
            "thumbnail_url": thumbnail_url,
            "event_count": db_info.get("event_count", 0),
            "total_duration": db_info.get("total_duration", 0.0),
            "processed": has_events,
        })

    return {"videos": result}


@app.get("/api/events/{video_id}")
def get_events(video_id: str):
    """Retrieve all structured events for a specific video from MongoDB."""
    mongo = get_mongo_manager()
    events = mongo.get_video_events(video_id)

    # Attach evidence thumbnails to each event
    for e in events:
        st = e.get("start_time", 0.0)
        et = e.get("end_time", 5.0)
        e["evidence_frames"] = _find_evidence_frames(video_id, st, et, max_frames=3)

    return {
        "video_id": video_id,
        "count": len(events),
        "events": events,
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = Query(True, description="Automatically process video after upload"),
):
    """Upload a new video file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    dest_path = settings.videos_upload_dir / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_id = dest_path.stem

    if auto_process:
        # Schedule pipeline in background
        background_tasks.add_task(run_pipeline_task, dest_path, video_id)
        return {
            "video_id": video_id,
            "filename": file.filename,
            "url": f"/data/videos/uploads/{file.filename}",
            "status": "processing_started",
            "message": "Video uploaded. Background processing pipeline started.",
        }

    return {
        "video_id": video_id,
        "filename": file.filename,
        "url": f"/data/videos/uploads/{file.filename}",
        "status": "uploaded",
        "message": "Video uploaded successfully.",
    }


PROCESSING_JOBS: Dict[str, Dict[str, Any]] = {}


@app.post("/api/process/{video_id}")
async def trigger_process_video(video_id: str, background_tasks: BackgroundTasks):
    """Trigger the video understanding pipeline for an uploaded video."""
    # Find video file
    matched_video = None
    for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        cand = settings.videos_upload_dir / f"{video_id}{ext}"
        if cand.is_file():
            matched_video = cand
            break

    if not matched_video:
        raise HTTPException(status_code=404, detail=f"Video file for '{video_id}' not found in uploads")

    if video_id in PROCESSING_JOBS and PROCESSING_JOBS[video_id]["status"] == "running":
        return {"status": "already_running", "message": f"Video '{video_id}' is already being processed."}

    PROCESSING_JOBS[video_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting pipeline...",
    }

    background_tasks.add_task(_execute_pipeline_job, matched_video, video_id)
    return {
        "status": "started",
        "video_id": video_id,
        "message": f"Processing started for {matched_video.name}.",
    }


def _execute_pipeline_job(video_path: Path, video_id: str):
    logger.info("Starting pipeline execution job for %s", video_id)
    try:
        PROCESSING_JOBS[video_id] = {"status": "running", "message": "Extracting frames and running YOLO..."}
        process_video(video_path, video_id=video_id)
        
        # Sync to MongoDB
        mongo = get_mongo_manager()
        event_file = settings.events_output_dir / f"{video_id}_events.json"
        if event_file.is_file():
            mongo.import_event_file(event_file)
            
        PROCESSING_JOBS[video_id] = {"status": "completed", "message": "Video understanding complete. Events saved to MongoDB."}
        logger.info("Pipeline completed successfully for %s", video_id)
    except Exception as e:
        logger.error("Pipeline failed for %s: %s", video_id, e)
        PROCESSING_JOBS[video_id] = {"status": "failed", "message": str(e)}


@app.get("/api/status/{video_id}")
def get_processing_status(video_id: str):
    """Check processing status of a video."""
    # Also check if events are already in MongoDB
    mongo = get_mongo_manager()
    events = mongo.get_video_events(video_id)
    if events:
        return {"status": "completed", "event_count": len(events), "message": "Ready"}
        
    job = PROCESSING_JOBS.get(video_id, {"status": "idle", "message": "Not processed yet"})
    return job


@app.post("/api/ask")
def ask_question(req: AskRequest):
    """
    Temporal reasoning QA endpoint.
    Retrieves events from MongoDB within or around the user's playback time window,
    sends context to Groq LLM, and returns direct answer with evidence timestamps.
    """
    mongo = get_mongo_manager()
    all_events = mongo.get_video_events(req.video_id)
    if not all_events:
        # Check if local event file exists and sync
        event_file = settings.events_output_dir / f"{req.video_id}_events.json"
        if event_file.is_file():
            mongo.import_event_file(event_file)
            all_events = mongo.get_video_events(req.video_id)

    if not all_events:
        raise HTTPException(
            status_code=404,
            detail=f"No events found for video '{req.video_id}'. Please process the video first.",
        )

    # Filter events if user specified explicit time window
    relevant_events = all_events
    if req.time_range_start is not None and req.time_range_end is not None:
        window_events = mongo.get_events_for_timerange(
            req.video_id,
            req.time_range_start,
            req.time_range_end,
        )
        if window_events:
            relevant_events = window_events

    # Call Groq Reasoning Service
    groq = get_groq_service()
    llm_result = groq.answer_query(
        query=req.query,
        events=relevant_events,
        current_time=req.current_time,
        time_range_start=req.time_range_start,
        time_range_end=req.time_range_end,
    )

    # Attach evidence frame images for the returned timestamp range
    start_time = llm_result.get("start_time", 0.0)
    end_time = llm_result.get("end_time", 5.0)
    evidence_frames = _find_evidence_frames(req.video_id, start_time, end_time, max_frames=4)

    return {
        "video_id": req.video_id,
        "query": req.query,
        "answer": llm_result.get("answer"),
        "start_time": start_time,
        "end_time": end_time,
        "time_formatted": f"{int(start_time//60):02d}:{int(start_time%60):02d} — {int(end_time//60):02d}:{int(end_time%60):02d}",
        "confidence": llm_result.get("confidence", 0.9),
        "key_objects": llm_result.get("key_objects", []),
        "evidence_frames": evidence_frames,
    }


# --------------------------------------------------------------------------
# Static mounts
# --------------------------------------------------------------------------

# Serve data directory (videos, frames)
data_dir = settings.project_root / "data"
if data_dir.is_dir():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

# Serve frontend directory
frontend_dir = settings.project_root / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
