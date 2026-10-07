"""
Test Qwen2.5-VL local model loading and inference.

This script loads the model independently (no full pipeline required).

Usage:
    python scripts/test_vlm.py
    python scripts/test_vlm.py --image path/to/frame.jpg
    python scripts/test_vlm.py --image frame1.jpg --image frame2.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL model loading & inference")
    parser.add_argument("--image", "-i", action="append", default=[],
                        help="Image file(s) to send to the VLM (can repeat)")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Custom prompt (overrides default)")
    parser.add_argument("--gpu-layers", type=int, default=None,
                        help="Override number of GPU layers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    settings = get_settings()

    from backend.vlm.model import QwenVLModel

    print("\n=== QWEN2.5-VL MODEL TEST ===\n")
    print(f"Model  : {settings.qwen_model_path}")
    print(f"MMProj : {settings.qwen_mmproj_path}")
    print(f"GPU layers: {args.gpu_layers or settings.qwen_n_gpu_layers}")

    # Check files exist
    if not settings.qwen_model_path.is_file():
        print(f"\n❌ Model file not found: {settings.qwen_model_path}")
        sys.exit(1)
    if not settings.qwen_mmproj_path.is_file():
        print(f"\n❌ MMProj file not found: {settings.qwen_mmproj_path}")
        sys.exit(1)

    print("\n--- Loading model ---\n")
    t0 = time.time()

    vlm = QwenVLModel(
        n_gpu_layers=args.gpu_layers if args.gpu_layers is not None else None,
    )
    vlm.load()

    load_time = time.time() - t0
    print(f"[OK] Model loaded in {load_time:.1f}s\n")

    # --- Inference test ---
    prompt = args.prompt or "Describe what you see in this image in 2-3 sentences."
    if not args.image:
        prompt = args.prompt or "What is 2 + 2? Answer briefly."

    image_paths = [p for p in args.image if Path(p).is_file()]
    if args.image and not image_paths:
        print("[!] No valid image files found, running text-only test.\n")

    print(f"Prompt: {prompt}")
    if image_paths:
        print(f"Images: {image_paths}")

    print("\n--- Running inference ---\n")
    t1 = time.time()

    response = vlm.chat(
        system_prompt="You are a helpful assistant. Be concise.",
        user_prompt=prompt,
        image_paths=image_paths if image_paths else None,
    )

    infer_time = time.time() - t1
    print(f"Response ({infer_time:.1f}s):\n")
    print(f"  {response}\n")

    vlm.unload()
    print("[OK] VLM test complete.\n")


if __name__ == "__main__":
    main()
