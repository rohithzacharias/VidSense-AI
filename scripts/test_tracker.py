"""
Test BoT-SORT object tracking on a short video segment.

Usage:
    python scripts/test_tracker.py --video data/videos/uploads/test.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Test BoT-SORT tracking")
    parser.add_argument("--video", "-v", required=False, help="Path to test video")
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="Only process the first N seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    settings = get_settings()

    if not args.video:
        uploads_dir = settings.videos_upload_dir
        found_videos = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi")) + list(uploads_dir.glob("*.mkv"))
        if found_videos:
            args.video = str(found_videos[0])
            print(f"Auto-detected video in uploads: {args.video}")
        else:
            print("❌ No video specified and none found in data/videos/uploads/.")
            print("Please place a video file (e.g. test.mp4) in data/videos/uploads/ or pass --video path/to/video.mp4")
            sys.exit(1)

    from backend.detection.tracker import BoTSORTTracker

    print("\n=== BoT-SORT TRACKING TEST ===\n")
    print(f"Video: {args.video}")
    print(f"Processing first {args.max_seconds}s at {settings.process_fps} FPS\n")

    tracker = BoTSORTTracker()

    # Use direct video tracking for consistent IDs
    tracking = tracker.track_video_file(
        video_path=args.video,
        process_fps=settings.process_fps,
    )

    # Filter to max_seconds
    tracking = [ft for ft in tracking if ft.timestamp <= args.max_seconds]

    print(f"\nTracked {len(tracking)} frames:\n")

    for ft in tracking:
        objs = ft.tracked_objects
        if objs:
            ids_str = ", ".join(
                f"{o.class_name} #{o.track_id} ({o.confidence:.2f})"
                for o in objs
            )
            print(f"  {ft.timestamp:6.2f}s → {ids_str}")
        else:
            print(f"  {ft.timestamp:6.2f}s → (no tracked objects)")

    # Summary
    summary = tracker.summarize_tracks(tracking)
    print(f"\n--- Track Summary ({len(summary)} unique IDs) ---\n")
    for tid, info in sorted(summary.items()):
        print(
            f"  Track #{tid}: {info['class_name']}  "
            f"seen {info['first_seen']:.1f}s–{info['last_seen']:.1f}s  "
            f"({info['frame_count']} frames)"
        )

    print("\n✓ Tracker test complete.\n")


if __name__ == "__main__":
    main()
