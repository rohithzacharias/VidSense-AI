"""
Complete video processing pipeline.

Orchestrates:
  Video → metadata → frame extraction → YOLO → BoT-SORT
        → temporal clips → VLM → event creation → JSON output

Can be run as a module:
    python -m backend.pipeline.process_video --video data/videos/uploads/test.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from backend.config.settings import get_settings, PROJECT_ROOT
from backend.video.metadata import get_video_metadata, VideoMetadata
from backend.video.extractor import extract_frames, ExtractedFrame
from backend.video.clips import create_temporal_clips, select_vlm_frames, TemporalClip
from backend.detection.tracker import BoTSORTTracker, FrameTracking
from backend.events.schemas import Event, VideoEvents
from backend.events.event_engine import EventEngine
from backend.vlm.model import QwenVLModel
from backend.vlm.inference import run_vlm_on_clip

logger = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _step(num: int, total: int, msg: str) -> None:
    """Print a pipeline progress step."""
    banner = f"\n{'='*60}\n[{num}/{total}] {msg}\n{'='*60}"
    print(banner)
    logger.info("[%d/%d] %s", num, total, msg)


def process_video(
    video_path: str | Path,
    video_id: Optional[str] = None,
    skip_vlm: bool = False,
) -> VideoEvents:
    """
    Run the full video understanding pipeline.

    Parameters
    ----------
    video_path : path to the input video.
    video_id : identifier for the video (auto-generated from filename if None).
    skip_vlm : if True, skip the VLM stage (useful for testing detection only).

    Returns
    -------
    VideoEvents containing all generated events.
    """
    settings = get_settings()
    settings.ensure_dirs()

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if video_id is None:
        video_id = video_path.stem

    total_steps = 5 if not skip_vlm else 3
    t0 = time.time()

    # ------------------------------------------------------------------
    # STEP 1 — Read video metadata
    # ------------------------------------------------------------------
    _step(1, total_steps, "Reading video metadata …")
    meta = get_video_metadata(video_path)
    print(f"  Resolution : {meta.width}×{meta.height}")
    print(f"  FPS        : {meta.fps}")
    print(f"  Duration   : {meta.duration:.1f}s ({meta.frame_count} frames)")
    print(f"  Codec      : {meta.codec}")

    # ------------------------------------------------------------------
    # STEP 2 — Extract frames at configured FPS
    # ------------------------------------------------------------------
    _step(2, total_steps, f"Extracting frames at {settings.process_fps} FPS …")
    frames = extract_frames(video_path, process_fps=settings.process_fps)
    print(f"  Extracted {len(frames)} frames")

    # ------------------------------------------------------------------
    # STEP 3 — YOLO + BoT-SORT tracking
    # ------------------------------------------------------------------
    _step(3, total_steps, "Running YOLO + BoT-SORT tracking …")
    tracker = BoTSORTTracker()
    tracking_data = tracker.track_video_file(
        video_path=video_path,
        process_fps=settings.process_fps,
    )

    # Save tracking results
    tracking_out = settings.tracking_output_dir / f"{video_id}_tracking.json"
    tracker.save_tracking(tracking_data, tracking_out)

    # Print track summary
    track_summary = tracker.summarize_tracks(tracking_data)
    print(f"  Tracked {len(tracking_data)} frames → {len(track_summary)} unique IDs")
    for tid, info in sorted(track_summary.items()):
        print(
            f"    Track #{tid}: {info['class_name']} — "
            f"{_fmt_time(info['first_seen'])}–{_fmt_time(info['last_seen'])} "
            f"({info['frame_count']} frames)"
        )

    if skip_vlm:
        print("\n⚠ VLM stage skipped (--skip-vlm flag).")
        # Create events with detection info only
        clips = create_temporal_clips(frames, meta.duration)
        engine = EventEngine(video_id)
        events: List[Event] = []
        for clip in clips:
            event = engine.create_event(
                clip=clip,
                vlm_description="[VLM skipped — detection-only mode]",
                tracking_data=tracking_data,
            )
            events.append(event)

        video_events = VideoEvents(
            video_id=video_id,
            source_video=str(video_path),
            total_duration=meta.duration,
            events=events,
        )
        out_path = settings.events_output_dir / f"{video_id}_events.json"
        video_events.save(out_path)
        print(f"\n[OK] Events saved -> {out_path}")
        return video_events

    # ------------------------------------------------------------------
    # STEP 4 — Temporal clips → VLM inference
    # ------------------------------------------------------------------
    _step(4, total_steps, "Running Qwen2.5-VL on temporal clips …")
    clips = create_temporal_clips(frames, meta.duration)
    print(f"  Created {len(clips)} clips ({settings.clip_duration}s each)")

    # Select VLM frames for each clip
    for clip in clips:
        select_vlm_frames(clip)
        print(
            f"  Clip {clip.clip_index}: {_fmt_time(clip.start_time)}–"
            f"{_fmt_time(clip.end_time)} — "
            f"{len(clip.all_frames)} frames, {len(clip.vlm_frames)} for VLM"
        )

    # Load VLM
    vlm = QwenVLModel()
    vlm.load()

    engine = EventEngine(video_id)
    events: List[Event] = []

    try:
        for i, clip in enumerate(clips):
            if not clip.vlm_frames:
                logger.warning("Clip %d has no frames — skipping", clip.clip_index)
                continue

            print(f"\n  ▶ Analysing clip {i+1}/{len(clips)} "
                  f"[{_fmt_time(clip.start_time)}–{_fmt_time(clip.end_time)}] …")

            # Run VLM
            description = run_vlm_on_clip(
                model=vlm,
                clip=clip,
                tracking_data=tracking_data,
            )

            # Create event
            event = engine.create_event(
                clip=clip,
                vlm_description=description,
                tracking_data=tracking_data,
            )
            events.append(event)

            # Print event
            print(f"  [EVENT] {event.format_time_range()}")
            print(f"    Type   : {event.event_type}")
            print(f"    Tracks : {event.track_ids}")
            print(f"    Conf   : {event.confidence:.3f}")
            print(f"    → {description[:200]}")
    finally:
        vlm.unload()

    # ------------------------------------------------------------------
    # STEP 5 — Save events
    # ------------------------------------------------------------------
    _step(5, total_steps, "Saving events …")
    video_events = VideoEvents(
        video_id=video_id,
        source_video=str(video_path),
        total_duration=meta.duration,
        events=events,
    )

    out_path = settings.events_output_dir / f"{video_id}_events.json"
    video_events.save(out_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"[OK] Pipeline complete in {elapsed:.1f}s")
    print(f"  Video    : {video_path.name}")
    print(f"  Events   : {len(events)}")
    print(f"  Output   : {out_path}")
    print(f"{'='*60}")

    # Print event timeline
    print("\n📋 EVENT TIMELINE\n")
    for e in events:
        print(f"  ● {e.format_time_range()}  [{e.event_type}]")
        print(f"    {e.description[:120]}")
        print()

    return video_events


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VidSense AI — Video Understanding Pipeline",
    )
    parser.add_argument(
        "--video", "-v",
        required=False,
        default=None,
        help="Path to the input video file (defaults to first video in data/videos/uploads).",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Custom video identifier (defaults to filename stem).",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM inference (test detection pipeline only).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override processing FPS (default: from settings).",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=None,
        help="Override clip duration in seconds (default: from settings).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    args = parser.parse_args()

    # Auto-detect video if not specified
    if not args.video:
        from backend.config.settings import get_settings
        _s = get_settings()
        uploads = _s.videos_upload_dir
        found = list(uploads.glob("*.mp4")) + list(uploads.glob("*.avi")) + list(uploads.glob("*.mkv"))
        if found:
            args.video = str(found[0])
            print(f"Auto-detected video in uploads: {args.video}\n")
        else:
            print("❌ No video specified and none found in data/videos/uploads/.")
            print("Please place a video file (e.g. test.mp4) in data/videos/uploads/ or pass --video path/to/video.mp4")
            sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Apply CLI overrides via environment variables (picked up by settings)
    import os
    if args.fps is not None:
        os.environ["VIDSENSE_PROCESS_FPS"] = str(args.fps)
    if args.clip_duration is not None:
        os.environ["VIDSENSE_CLIP_DURATION"] = str(args.clip_duration)

    try:
        process_video(
            video_path=args.video,
            video_id=args.video_id,
            skip_vlm=args.skip_vlm,
        )
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
