# 🎥 VidSense AI — Video Understanding & Temporal Reasoning

> **AI-powered video understanding platform that transforms raw surveillance and CCTV video feeds into structured, queryable semantic events with precise timestamp citations, object tracking, and evidence frames.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-FF6F00.svg)](https://docs.ultralytics.com/)
[![Qwen2.5-VL](https://img.shields.io/badge/VLM-Qwen2.5--VL--3B-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF)
[![Groq](https://img.shields.io/badge/Reasoning-Groq_LLM-orange.svg)](https://groq.com/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-green.svg)](https://www.mongodb.com/)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Folder Structure](#-folder-structure)
- [Hardware & Software Prerequisites](#-hardware--software-prerequisites)
- [Complete Installation Guide](#-complete-installation-guide)
  - [1. Clone Repository & Setup Virtual Environment](#1-clone-repository--setup-virtual-environment)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Install PyTorch with CUDA](#3-install-pytorch-with-cuda)
  - [4. Install llama-cpp-python (with CUDA support)](#4-install-llama-cpp-python-with-cuda-support)
  - [5. Download Qwen2.5-VL Model Files](#5-download-qwen25-vl-model-files)
  - [6. Configure Environment Variables (.env)](#6-configure-environment-variables-env)
  - [7. Start MongoDB](#7-start-mongodb)
- [Running the Project](#-running-the-project)
  - [Option 1: Launch Web Application & API (Recommended)](#option-1-launch-web-application--api-recommended)
  - [Option 2: Run Pipeline via Command Line (CLI)](#option-2-run-pipeline-via-command-line-cli)
- [Testing Individual Components](#-testing-individual-components)
- [REST API Reference](#-rest-api-reference)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Overview

VidSense AI bridges traditional computer vision with modern multimodal LLMs to solve complex video reasoning questions:
- **Frame Sampling & Metadata Extraction**: Samples frames at configurable FPS (default: 10 FPS).
- **Object Detection & Multi-Object Tracking**: YOLOv8 + BoT-SORT maintain persistent entity IDs (e.g. `Person #1`, `Car #2`) across frames.
- **Temporal Window Partitioning**: Slices video streams into 5-second semantic clips with keyframes.
- **Local Vision-Language Reasoning**: Qwen2.5-VL 3B GGUF generates rich semantic event descriptions locally on GPU.
- **Persistent Structured Event Store**: Stores normalized event timelines in MongoDB (`localhost:27017`).
- **Natural Language Question Answering**: Fast temporal reasoning with Groq LLM to locate timestamps and cite evidence frames.
- **Glassmorphic Web Dashboard**: Full interactive interface with synchronized video playback, scrubbing, Q&A chat, and live timeline event cards.

---

## 🏗️ System Architecture

```
                                      [ Input Video (.mp4) ]
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  10 FPS Frame Sample  │
                                    └───────────┬───────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        ┌─────────────────────────┐                           ┌─────────────────────────┐
        │   YOLOv8 Detection      │                           │  5s Temporal Windows    │
        │   + BoT-SORT Tracking   │                           │  (6-8 Keyframes/Clip)   │
        └────────────┬────────────┘                           └────────────┬────────────┘
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │   Qwen2.5-VL 3B (Local)     │
                                 │   Clip-level VLM Inference  │
                                 └──────────────┬──────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │       Event Engine          │
                                 │  Normalized JSON Generation │
                                 └──────────────┬──────────────┘
                                                ▼
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
          ┌─────────────────────────┐                       ┌─────────────────────────┐
          │     MongoDB Store       │                       │ outputs/events/*.json   │
          └────────────┬────────────┘                       └─────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │    Groq Reasoning LLM   │◄───────── [ User Query: "When did the car stop?" ]
          └────────────┬────────────┘
                       │
                       ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │  Web Dashboard: Playhead Sync + Event Timeline + Evidence Snapshots    │
          └────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
Hacknex-VidSense AI/
├── backend/
│   ├── api/
│   │   └── app.py               # FastAPI server, REST routes, static file serving
│   ├── config/
│   │   └── settings.py          # Centralized configuration & environment loader
│   ├── database/
│   │   └── mongo.py             # MongoDB client, schema upserts & range queries
│   ├── detection/
│   │   ├── yolo.py              # YOLOv8 object detector
│   │   └── tracker.py           # BoT-SORT multi-object tracker
│   ├── events/
│   │   ├── schemas.py           # Pydantic data schemas for Events & Videos
│   │   └── event_engine.py      # Transforms raw VLM & tracking into structured Events
│   ├── llm/
│   │   └── groq_service.py      # Groq LLM integration for temporal reasoning Q&A
│   ├── pipeline/
│   │   └── process_video.py     # End-to-end video processing pipeline orchestrator
│   ├── video/
│   │   ├── metadata.py          # Video resolution, duration & FPS probe
│   │   ├── extractor.py         # OpenCV frame extraction at fixed FPS
│   │   └── clips.py             # Temporal clip window generator
│   └── vlm/
│       ├── model.py             # Local Qwen2.5-VL loader (llama-cpp-python)
│       ├── inference.py         # Multi-frame VLM inference engine
│       └── prompts.py           # Structured system and user prompt templates
│
├── frontend/
│   ├── index.html               # Web UI layout
│   ├── style.css                # Glassmorphic dark styling & animations
│   └── app.js                   # UI logic, video player sync, API interaction
│
├── models/
│   └── qwen/                    # Local GGUF models (place weights here)
│       ├── Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
│       └── Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf
│
├── data/
│   ├── videos/
│   │   ├── uploads/             # Raw uploaded video files (.mp4)
│   │   └── processed/           # Processed video output
│   ├── frames/                  # Extracted frame caches
│   └── clips/                   # Temporal clip keyframes
│
├── outputs/
│   ├── events/                  # Persisted event JSON output
│   ├── detections/              # Raw detection metadata
│   └── tracking/                # Tracking trajectories
│
├── scripts/
│   ├── test_video.py            # Diagnostic script for frame extraction
│   ├── test_yolo.py             # Diagnostic script for YOLO detections
│   ├── test_tracker.py          # Diagnostic script for BoT-SORT tracking
│   └── test_vlm.py              # Diagnostic script for Qwen2.5-VL inference
│
├── .env.example                 # Configuration template
├── .gitignore                   # Excludes weights, large videos & secrets
├── requirements.txt             # Pip package dependencies
└── pyproject.toml               # Project metadata & build configuration
```

---

## 💻 Hardware & Software Prerequisites

| Component | Minimum Specification | Recommended Specification |
|:---|:---|:---|
| **OS** | Windows 10/11 or Ubuntu 22.04+ | Windows 11 / Ubuntu 22.04 |
| **GPU** | NVIDIA GTX 1660 / RTX 3050 (4 GB VRAM) | NVIDIA RTX 3060 / 4060 (6–8 GB+ VRAM) |
| **RAM** | 16 GB | 32 GB |
| **Python** | Python 3.10 or 3.11 | Python 3.10 |
| **CUDA** | CUDA 12.1+ | CUDA 12.1 or 12.4 |
| **Database** | MongoDB 6.0+ (Local or Docker) | MongoDB 7.0+ |
| **Tool** | [uv](https://docs.astral.sh/uv/) (recommended) or `pip` | `uv` package manager |

---

## 🚀 Complete Installation Guide

### 1. Clone Repository & Setup Virtual Environment

Open PowerShell (Windows) or Terminal (Linux/macOS):

```bash
# Clone the repository
git clone https://github.com/rohithzacharias/VidSense-AI.git
cd VidSense-AI

# Create virtual environment with uv (fastest)
uv venv .venv --python 3.10

# Activate virtual environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

*(Alternatively with standard python: `python -m venv .venv`)*

---

### 2. Install Dependencies

Install all core dependencies including OpenCV, YOLO, FastAPI, PyMongo, Groq, and Pydantic:

```bash
uv pip install -r requirements.txt
# OR with pip:
# pip install -r requirements.txt
```

---

### 3. Install PyTorch with CUDA

Install PyTorch compiled with CUDA 12.1:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### 4. Install llama-cpp-python (with CUDA support)

`llama-cpp-python` runs Qwen2.5-VL locally on your GPU.

#### Option A: Pre-built CUDA Wheel (Recommended for Windows / CUDA 12.4)
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

#### Option B: Build with CUDA from source
```powershell
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
uv pip install llama-cpp-python --no-binary llama-cpp-python
```

#### Option C: CPU-only Fallback (if no GPU is available)
```bash
uv pip install llama-cpp-python
```

---

### 5. Download Qwen2.5-VL Model Files

VidSense AI uses the quantized **Qwen2.5-VL-3B-Instruct** model.

Create the target directory:
```bash
mkdir -p models/qwen
```

Download both required files from Hugging Face into `models/qwen/`:

1. **Main Model Weights (Q4_K_M)**:
   - File: `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (~1.93 GB)
   - Source: [Qwen/Qwen2.5-VL-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF/blob/main/qwen2.5-vl-3b-instruct-q4_k_m.gguf)

2. **Multimodal Projector (mmproj-f16)**:
   - File: `Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf` (~1.34 GB)
   - Source: [Qwen/Qwen2.5-VL-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF/blob/main/qwen2.5-vl-3b-instruct-mmproj-f16.gguf)

#### Download via Hugging Face CLI:
```bash
uv pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct-GGUF qwen2.5-vl-3b-instruct-q4_k_m.gguf --local-dir models/qwen --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct-GGUF qwen2.5-vl-3b-instruct-mmproj-f16.gguf --local-dir models/qwen --local-dir-use-symlinks False
```

Ensure the filenames inside `models/qwen/` match:
- `models/qwen/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`
- `models/qwen/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf`

---

### 6. Configure Environment Variables (.env)

Copy the configuration template:

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```ini
# --- Frame Sampling & Video ---
VIDSENSE_PROCESS_FPS=10
VIDSENSE_CLIP_DURATION=5
VIDSENSE_VLM_FRAMES_PER_CLIP=6

# --- YOLO Detection ---
VIDSENSE_YOLO_MODEL=yolov8n.pt
VIDSENSE_YOLO_CONFIDENCE=0.35
VIDSENSE_YOLO_DEVICE=0

# --- Local Qwen VLM GPU Offload ---
# Set to 20 for ~2.5GB VRAM, -1 for full GPU (~4GB+), or 0 for CPU-only
VIDSENSE_QWEN_GPU_LAYERS=20
VIDSENSE_QWEN_CTX_SIZE=8192
VIDSENSE_QWEN_MAX_TOKENS=512

# --- Groq LLM Temporal Reasoning ---
# Get free API key from https://console.groq.com/
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# --- MongoDB Database ---
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=vidsense_ai
```

---

### 7. Start MongoDB

VidSense AI uses MongoDB to store and query timestamped events.

#### Option A: Docker (Fastest)
```bash
docker run -d -p 27017:27017 --name vidsense-mongo mongo:latest
```

#### Option B: Local Windows Service
If you installed MongoDB Community Server, ensure the service is running:
```powershell
net start MongoDB
```

Verify connection:
```bash
python -c "from backend.database.mongo import get_mongo_manager; print('MongoDB connected:', get_mongo_manager().is_connected())"
```

---

## 🏃 Running the Project

### Option 1: Launch Web Application & API (Recommended)

Start the unified FastAPI server:

```bash
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Once started:
1. Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)**.
2. **Upload a video** (`.mp4`) via the UI or select a pre-loaded sample.
3. Click **⚡ Process Video** to run YOLO tracking, temporal clip extraction, and Qwen VLM inference.
4. Interact with the **Ask AI** query box:
   - *"When did the person appear?"*
   - *"What happened around 00:15?"*
   - *"How many cars drove by?"*
5. Click on timeline events or citations to seek the video player directly to that second!

---

### Option 2: Run Pipeline via Command Line (CLI)

You can process any video file directly using the CLI orchestrator:

```bash
python -m backend.pipeline.process_video --video data/videos/uploads/skating.mp4
```

#### CLI Options & Overrides:

```bash
python -m backend.pipeline.process_video \
  --video data/videos/uploads/test.mp4 \
  --video-id sample_run_01 \
  --fps 10 \
  --clip-duration 5 \
  --skip-vlm \
  --log-level INFO
```

| Argument | Description | Default |
|:---|:---|:---|
| `--video` | Path to input video file (Required) | — |
| `--video-id` | Unique ID for database indexing | Video filename stem |
| `--fps` | Sampling frame rate | `10.0` |
| `--clip-duration` | Seconds per temporal window | `5.0` |
| `--skip-vlm` | Run only YOLO + BoT-SORT (skip Qwen) | `False` |
| `--log-level` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

Structured output will automatically be saved to `outputs/events/<video_id>_events.json` and synced to MongoDB.

---

## 🧪 Testing Individual Components

The `scripts/` directory contains standalone diagnostic scripts to verify each pipeline stage in isolation:

```bash
# 1. Test video reader & frame extraction
python scripts/test_video.py --video data/videos/uploads/skating.mp4

# 2. Test YOLOv8 object detection
python scripts/test_yolo.py --video data/videos/uploads/skating.mp4

# 3. Test BoT-SORT multi-object tracking (persistent track IDs)
python scripts/test_tracker.py --video data/videos/uploads/skating.mp4

# 4. Test Qwen2.5-VL model inference (text or image)
python scripts/test_vlm.py
python scripts/test_vlm.py --image data/frames/skating/frame_000001.jpg
```

---

## 🌐 REST API Reference

When the FastAPI server is running, interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves the interactive frontend UI |
| `GET` | `/api/health` | Server health and MongoDB connection status |
| `GET` | `/api/videos` | Returns list of all available and processed videos |
| `POST` | `/api/upload` | Upload a new `.mp4` video file |
| `POST` | `/api/process` | Trigger asynchronous pipeline execution for a video |
| `GET` | `/api/events/{video_id}` | Retrieve all structured events stored for a video |
| `POST` | `/api/ask` | Natural language question answering with temporal citations |

#### Example `/api/ask` Request Body:
```json
{
  "video_id": "skating",
  "query": "Where did the skateboarder fall?",
  "current_time": 12.5,
  "time_range_start": 0.0,
  "time_range_end": 30.0
}
```

#### Example `/api/ask` Response:
```json
{
  "video_id": "skating",
  "query": "Where did the skateboarder fall?",
  "answer": "Person #1 attempted a ramp jump at 00:14.2 and lost balance, landing at 00:16.8.",
  "start_time": 14.2,
  "end_time": 16.8,
  "time_formatted": "00:14 — 00:16",
  "confidence": 0.94,
  "key_objects": ["person", "skateboard"],
  "evidence_frames": [
    "/data/clips/skating/clip_000003_frame_000142.jpg"
  ]
}
```

---

## ⚙️ Configuration Reference

All settings are managed via `backend/config/settings.py` and can be customized in your `.env` file:

| Variable | Default | Description |
|:---|:---|:---|
| `VIDSENSE_PROCESS_FPS` | `10` | Frame extraction frequency (FPS) |
| `VIDSENSE_CLIP_DURATION` | `5` | Duration of each temporal window (seconds) |
| `VIDSENSE_VLM_FRAMES_PER_CLIP` | `6` | Number of keyframes sent to Qwen VLM per clip |
| `VIDSENSE_YOLO_MODEL` | `yolov8n.pt` | Ultralytics model checkpoint (auto-downloaded) |
| `VIDSENSE_YOLO_CONFIDENCE` | `0.35` | Minimum detection confidence threshold |
| `VIDSENSE_YOLO_DEVICE` | `0` | GPU device index (`0`, `1`) or `"cpu"` |
| `VIDSENSE_TRACKER` | `botsort.yaml` | Tracker algorithm configuration |
| `VIDSENSE_QWEN_GPU_LAYERS` | `20` | Layers offloaded to GPU (`0` = CPU, `-1` = all) |
| `VIDSENSE_QWEN_CTX_SIZE` | `8192` | Qwen context window token length |
| `VIDSENSE_QWEN_MAX_TOKENS` | `512` | Max generation token length |
| `GROQ_API_KEY` | — | Groq Cloud API key for reasoning |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model identifier |
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection URI |
| `MONGO_DB` | `vidsense_ai` | MongoDB database name |
| `VIDSENSE_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

---

## 🛠️ Troubleshooting & FAQs

### 1. `CUDA not available` after installing PyTorch
- Verify your NVIDIA drivers via `nvidia-smi`.
- Ensure you installed the CUDA-enabled PyTorch wheel:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### 2. `llama-cpp-python` fails to build or load with CUDA
- Use the prebuilt CUDA wheel rather than compiling from source:
  ```bash
  uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
  ```
- If building on Windows, ensure Visual Studio C++ Build Tools and CMake are installed.

### 3. Out of GPU Memory (CUDA VRAM Error)
- Reduce `VIDSENSE_QWEN_GPU_LAYERS` in `.env` to `15` or `10`.
- Reduce `VIDSENSE_VLM_FRAMES_PER_CLIP` from `6` to `4`.
- If testing detection only, pass `--skip-vlm` to bypass the VLM stage entirely.

### 4. `Form data requires "python-multipart"` error on upload
- Run `uv pip install python-multipart` (already included in the updated `requirements.txt`).

### 5. MongoDB connection warning
- If MongoDB is not running locally, start it with Docker:
  ```bash
  docker run -d -p 27017:27017 mongo:latest
  ```

---

## 📄 License

This project is licensed under the MIT License.
