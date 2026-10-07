"""
Centralized configuration for the VidSense AI video processing pipeline.

All configurable values live here. Override via environment variables or .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: backend/config/ → root)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent


def _env(key: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


@dataclass(frozen=True)
class Settings:
    """Immutable configuration snapshot for the pipeline."""

    # ── Project paths ─────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    # Video I/O
    videos_upload_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "videos" / "uploads"
    )
    videos_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "videos" / "processed"
    )
    frames_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "frames"
    )
    clips_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "clips"
    )

    # Output directories
    detections_output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "detections"
    )
    tracking_output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "tracking"
    )
    events_output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "events"
    )

    # ── Frame extraction ──────────────────────────────────────────────────
    process_fps: float = field(
        default_factory=lambda: _env_float("VIDSENSE_PROCESS_FPS", 10.0)
    )

    # ── Temporal clips ────────────────────────────────────────────────────
    clip_duration: float = field(
        default_factory=lambda: _env_float("VIDSENSE_CLIP_DURATION", 5.0)
    )
    vlm_frames_per_clip: int = field(
        default_factory=lambda: _env_int("VIDSENSE_VLM_FRAMES_PER_CLIP", 6)
    )

    # ── YOLO ──────────────────────────────────────────────────────────────
    yolo_model: str = field(
        default_factory=lambda: _env("VIDSENSE_YOLO_MODEL", "yolov8n.pt")
    )
    yolo_confidence: float = field(
        default_factory=lambda: _env_float("VIDSENSE_YOLO_CONFIDENCE", 0.35)
    )
    yolo_device: str = field(
        default_factory=lambda: _env("VIDSENSE_YOLO_DEVICE", "0")  # "0" = first GPU, "cpu" for CPU
    )

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker_type: str = field(
        default_factory=lambda: _env("VIDSENSE_TRACKER", "botsort.yaml")
    )

    # ── Qwen2.5-VL ───────────────────────────────────────────────────────
    qwen_model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "models"
        / "qwen"
        / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    )
    qwen_mmproj_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "models"
        / "qwen"
        / "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"
    )
    # Number of layers to offload to GPU (0 = CPU-only, -1 = all to GPU)
    qwen_n_gpu_layers: int = field(
        default_factory=lambda: _env_int("VIDSENSE_QWEN_GPU_LAYERS", 20)
    )
    qwen_ctx_size: int = field(
        default_factory=lambda: _env_int("VIDSENSE_QWEN_CTX_SIZE", 8192)
    )
    qwen_max_tokens: int = field(
        default_factory=lambda: _env_int("VIDSENSE_QWEN_MAX_TOKENS", 512)
    )
    qwen_temperature: float = field(
        default_factory=lambda: _env_float("VIDSENSE_QWEN_TEMPERATURE", 0.2)
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: _env("VIDSENSE_LOG_LEVEL", "INFO")
    )

    # ── Helpers ───────────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in (
            self.videos_upload_dir,
            self.videos_processed_dir,
            self.frames_dir,
            self.clips_dir,
            self.detections_output_dir,
            self.tracking_output_dir,
            self.events_output_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return (and cache) the global settings instance."""
    # Attempt to load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    return Settings()
