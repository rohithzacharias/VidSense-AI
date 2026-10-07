"""
Test script for video metadata and frame extraction.

Usage:
    python scripts/test_video.py --video data/videos/uploads/test.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.video.metadata import get_video_metadata
from backend.video.extractor import extract_frames
from backend.video.clips import create_temporal_clips, select_vlm_frames
from backend.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Test video metadata + frame extraction")
    parser.add_argument("--video", "-v", required=False, help="Path to test video")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    settings = get_settings()
    settings.ensure_dirs()

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

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    # --- Metadata ---
    print("\n=== VIDEO METADATA ===\n")
    meta = get_video_metadata(video_path)
    for k, v in meta.to_dict().items():
        print(f"  {k:15s}: {v}")

    # --- Frame extraction ---
    print(f"\n=== FRAME EXTRACTION ({settings.process_fps} FPS) ===\n")
    frames = extract_frames(video_path, process_fps=settings.process_fps)
    print(f"  Total extracted: {len(frames)} frames")

    if frames:
        print(f"  First frame : idx={frames[0].frame_index}  ts={frames[0].timestamp:.3f}s")
        print(f"  Last frame  : idx={frames[-1].frame_index}  ts={frames[-1].timestamp:.3f}s")

        # Verify timestamps are monotonically increasing
        for i in range(1, len(frames)):
            if frames[i].timestamp <= frames[i - 1].timestamp:
                print(f"  ⚠ Timestamp not increasing at index {i}!")
                break
        else:
            print("  ✓ All timestamps monotonically increasing")

        # Verify frame files exist
        missing = [f for f in frames if f.path and not Path(f.path).is_file()]
        if missing:
            print(f"  ⚠ {len(missing)} frame files missing on disk!")
        else:
            print(f"  ✓ All {len(frames)} frame files saved to disk")

    # --- Temporal clips ---
    print(f"\n=== TEMPORAL CLIPS ({settings.clip_duration}s windows) ===\n")
    clips = create_temporal_clips(frames, meta.duration)
    print(f"  Total clips: {len(clips)}")

    for clip in clips:
        select_vlm_frames(clip)
        print(
            f"  Clip {clip.clip_index:3d}: "
            f"{clip.start_time:6.1f}s – {clip.end_time:6.1f}s  "
            f"| {len(clip.all_frames):3d} frames  "
            f"| {len(clip.vlm_frames):2d} VLM frames"
        )

    print("\n✓ Video test complete.\n")


if __name__ == "__main__":
    main()
