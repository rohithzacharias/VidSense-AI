"""
Test YOLO object detection on a few frames.

Usage:
    python scripts/test_yolo.py --video data/videos/uploads/test.mp4
    python scripts/test_yolo.py --image path/to/frame.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YOLO object detection")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--video", "-v", help="Path to a video (will extract a few frames)")
    group.add_argument("--image", "-i", help="Path to a single image")
    parser.add_argument("--frames", type=int, default=5, help="Number of test frames from video")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import cv2
    import numpy as np
    from backend.detection.yolo import YOLODetector

    settings = get_settings()
    detector = YOLODetector()
    print("\n=== YOLO DETECTION TEST ===\n")

    if not args.image and not args.video:
        # Check if any videos exist in uploads directory
        uploads_dir = settings.videos_upload_dir
        found_videos = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi")) + list(uploads_dir.glob("*.mkv"))
        if found_videos:
            args.video = str(found_videos[0])
            print(f"Auto-detected video in uploads: {args.video}\n")
        else:
            print("No image/video specified and none found in uploads. Generating a synthetic test frame...")
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_img[:] = (240, 240, 240)
            cv2.rectangle(test_img, (100, 150), (250, 450), (50, 50, 200), -1)
            cv2.putText(test_img, "Test Object", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            fd = detector.detect_frame(test_img, timestamp=0.0, frame_index=0)
            print(f"YOLO model loaded and ran successfully on test image.")
            print(f"Detections found: {len(fd.detections)}")
            for d in fd.detections:
                print(f"  {d.class_name} (conf={d.confidence:.2f}) bbox={d.bbox}")
            print("\nTip: To test on real data, pass --video path/to/video.mp4")
            print("\n✓ YOLO test complete.\n")
            return

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ Cannot read image: {args.image}")
            sys.exit(1)

        fd = detector.detect_frame(img, timestamp=0.0, frame_index=0)
        print(f"Detections on {args.image}:")
        for d in fd.detections:
            print(f"  {d.class_name} (conf={d.confidence:.2f}) bbox={d.bbox}")
        if not fd.detections:
            print("  (no detections)")
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {args.video}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total // args.frames)

        print(f"Video: {args.video} ({total} frames, {fps:.1f} FPS)")
        print(f"Testing {args.frames} evenly-spaced frames:\n")

        count = 0
        frame_num = 0
        while count < args.frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % interval == 0:
                ts = round(frame_num / fps, 2)
                fd = detector.detect_frame(frame, timestamp=ts, frame_index=frame_num)
                print(f"  Frame {frame_num} ({ts:.1f}s) → {len(fd.detections)} detections")
                for d in fd.detections:
                    print(f"    {d.class_name} conf={d.confidence:.2f} bbox={d.bbox}")
                count += 1
            frame_num += 1

        cap.release()

    print("\n✓ YOLO test complete.\n")


if __name__ == "__main__":
    main()
