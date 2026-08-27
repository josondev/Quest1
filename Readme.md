# Quest1 — Dynamic Dialogue Detection Engine

An end-to-end media analysis system that locates target spoken or visual dialogue within local video files and remote streaming URLs.

Quest1 uses a progressive five-tier fallback pipeline to balance processing speed, computational cost, and detection accuracy. Instead of depending on a single modality, it combines subtitle matching, speech recognition, OCR, and Vision-Language Model verification to locate and validate dialogue occurrences.

---

## Overview

Given a video source and a target dialogue, Quest1 attempts to determine where the dialogue occurs and, when possible, identify the corresponding visual frame.

The system can work with:

- Local video files
- Remote video URLs
- CDN-hosted streams
- Adaptive streaming sources
- Standard MP4 sources

The core processing strategy is progressive — it starts with the cheapest available source of evidence and escalates only when necessary:

```
Video Source
     |
     v
Media Ingestion
     |
     v
Detection Pipeline
     |
     +-- Subtitles (Tier 0)
     +-- Audio      (Tier 1)
     +-- Visual      (Tier 2 -> Tier 3 -> Tier 4)
     |
     v
Detection / Uncertain
```

---

## Key Features

### Five-Tier Detection Pipeline

```
Fast + Low Compute
        |
        v
Tier 0  -> Embedded Subtitle Matching
        |
        v
Tier 1  -> Audio Speech-to-Text Alignment
        |
        v
Tier 2  -> Sparse OCR Search
        |
        v
Tier 3  -> Dense OCR Confirmation
        |
        v
Tier 4  -> Vision-Language Model Arbitration
        |
        v
Reliable Detection / Uncertain Result
```

This design allows the system to avoid expensive visual processing when subtitles or audio are sufficient to localize the target.

---

## Detection Pipeline

The main orchestration logic is implemented in `src/pipeline.py`.

**Tier 0 — Embedded Subtitle Matching**
Checks available subtitle/VTT data and attempts to locate the target using sequence similarity, providing a fast initial localization without speech recognition or visual processing.

**Tier 1 — Audio Speech-to-Text**
If subtitle-based localization is unavailable or insufficient, Quest1 extracts audio and uses local `faster-whisper` speech recognition with word-level timestamp alignment to find where the target dialogue is spoken.

**Tier 2 — Sparse OCR**
If the target still can't be reliably localized, frames are sampled across the video timeline and OCR searches for the target text — useful for text that appears visually but isn't spoken (title cards, signs, posters, overlays).

**Tier 3 — Dense OCR Confirmation**
Once a candidate region is identified, Quest1 performs concentrated OCR processing around that window to refine the detection and pinpoint the visual occurrence more precisely.

**Tier 4 — Vision-Language Model Arbitration**
The final tier evaluates multiple candidate frame windows (C1–C7) around a detected timestamp using a Vision-Language Model, weighing text visibility, visual context, spatial context, and candidate-frame consistency. This provides an additional verification layer when OCR alone isn't sufficient.

---

## Media Ingestion

Location: `src/ingestion.py`

This layer is responsible for probing media sources, extracting streams, handling remote URLs, and preparing audio/video data for downstream processing.

**CDN and Adaptive Stream Parsing** — Some streaming URLs don't follow conventional patterns (no `.m3u8` or `.mpd` extension). Quest1 detects these non-standard cases and can route extraction through a full MP4 download/cache path when required.

**Header Signature Injection** — Some CDN and streaming sources require browser-like request headers. Quest1 supports injecting headers such as `User-Agent` and `Referer` into FFmpeg and `yt-dlp` operations for sources with hotlink protection.

**Audio Codec Verification** — Before speech recognition, Quest1 verifies a usable audio stream exists and filters out video-only sources. The generated speech-recognition input is 16 kHz, mono, PCM WAV.

**Windows OpenCV Stream Handling** — Remote streams can behave unreliably with `cv2.VideoCapture(...)` on Windows. Quest1 avoids passing remote HTTP/HTTPS streams directly to OpenCV and instead routes remote frame extraction through FFmpeg seeking, avoiding `CAP_IMAGES` runtime errors.

**Local File Re-routing** — When a remote stream can't be processed directly and a download fallback triggers, Quest1 updates the active video source to the locally cached MP4 so all downstream stages operate against the same local file.

**OK.ru and Remote CDN Handling** — Remote-source failures (ISP restrictions, campus-network restrictions, connection failures, required headers, format differences) are distinguished from detection failures — a network/access failure does not mean the dialogue wasn't found. When a remote source can't be reached reliably, the practical fallback is to obtain the video locally and run Quest1 against the local file.

---

## Offline Speech Recognition

Location: `src/primary_stt.py`

Quest1 integrates local `faster-whisper` speech recognition, using available CPU/GPU resources and Hugging Face model weights. This avoids depending on a remote speech-recognition API, external rate limits, cloud payload limitations, and reduces reliance on network availability during inference.

The implementation also contains sliding-window word-alignment logic to prevent timestamp index drift when aligning recognized words with candidate dialogue windows.

---

## Vision-Language Arbitration Service

Location: `src/fallback_vlm.py`

Coordinates the Tier 4 multi-candidate VLM arbitration described above.

---

## Project Structure

```
Quest1/
├── artifacts/               # Generated job outputs, JSON metadata, and persistent frame images
├── src/
│   ├── app.py                # FastAPI REST endpoints and background task routing
│   ├── config.py              # System and environment configuration
│   ├── fallback_vlm.py         # Vision-Language Model candidate arbitration service
│   ├── ingestion.py            # Stream probing, audio extraction, and FFmpeg wrappers
│   ├── pipeline.py             # Five-tier PipelineOrchestrator implementation
│   ├── primary_ocr.py          # Sparse timeline and dense-window OCR scanners
│   ├── primary_stt.py          # Local faster-whisper STT service and alignment logic
│   └── models/
│       └── schemas.py           # Pydantic domain models
├── tests/                    # Test suite
├── frontend.py               # NiceGUI dashboard, calls into the FastAPI backend
├── Dockerfile
├── docker-compose.yml
├── architecture.md
├── pytest.ini
├── .env.example
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- FFmpeg
- Git
- pip

Verify:

```bash
python --version
ffmpeg -version
```

### Installing FFmpeg

**Windows** (via winget):
```bash
winget install --id Gyan.FFmpeg
```

**Ubuntu / Debian**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS** (via Homebrew):
```bash
brew install ffmpeg
```

### Installing Quest1

**1. Clone the repository**

```bash
git clone https://github.com/josondev/Quest1.git
cd Quest1
```

**2. Create a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
pip install faster-whisper nicegui
```

(If `faster-whisper` and `nicegui` are already pinned in `requirements.txt`, the second command isn't necessary.)

**4. Configure environment variables**

Copy `.env.example` to `.env` and fill in the values your setup needs (e.g. API keys for the VLM provider) before starting the backend.

---

## Running Quest1

### 1. Start the backend

```bash
uvicorn src.app:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

### 2. Start the dashboard

In a separate terminal:

```bash
python frontend.py
```

Open `http://127.0.0.1:8080` to submit detection jobs (by URL or local file), watch live job status, and view candidate/detected frames as they're produced.

### Running with Docker

A `Dockerfile` and `docker-compose.yml` are included for containerized runs:

```bash
docker compose up --build
```

Check `docker-compose.yml` for the exact ports and service configuration before relying on this path.

### Running detection directly against the API

If you want to submit a job without the dashboard, hit the FastAPI backend directly, e.g.:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_url": "VIDEO_URL", "text": "TARGET_DIALOGUE"}'
```

Then poll job status at `GET /api/v1/jobs/{job_id}`. Check `src/app.py` for the exact request/response schema, since field names may differ from the example above.

Using a local file is also the recommended fallback when a remote source such as OK.ru can't be reached from the current network — download it separately and point Quest1 at the local path instead of the URL.

---

## Output Artifacts

Quest1 stores generated job artifacts under `artifacts/`, which can include:

```
artifacts/
├── result.json
├── candidate frames
├── detected frames
└── metadata
```

Result data can include the detection timestamp, confidence score, pipeline tier used, extracted text, frame information, and candidate information. The exact artifact structure is determined by the current implementation.

---

## Spoken Dialogue vs. Visual Dialogue

A key design consideration: spoken dialogue does not necessarily appear visually.

```
Audio -> Speech Recognition -> Target Dialogue Located -> OCR Verification -> No Matching Text
```

Speech recognition can successfully locate dialogue while OCR finds no corresponding on-screen text. This happens when the dialogue is only spoken, there are no subtitles, there's no burned-in text, or the line simply isn't rendered visually. An audio timestamp should therefore not automatically be treated as proof of a visual frame — Quest1 keeps localization and visual confirmation as separate concepts.

---

## Design Philosophy

**Progressive Computation** — Start with inexpensive evidence (subtitles, then speech) and escalate to sparse OCR, dense OCR, and VLM verification only when required, avoiding expensive visual processing when an earlier stage already provides a suitable candidate.

**Evidence-Based Detection** — Quest1 distinguishes between "Detected" and "Uncertain." A speech-recognition timestamp alone isn't treated as a confirmed visual occurrence, and an OCR match is considered in its surrounding visual context rather than accepting a single frame blindly.

**Separation of Media Access and Detection** — Remote media retrieval is treated as a separate problem from dialogue detection, making it possible to distinguish "the video couldn't be accessed" from "the video was processed but the target couldn't be confirmed."

---

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Speech Recognition | faster-whisper |
| Media Processing | FFmpeg |
| Media Extraction | yt-dlp |
| OCR Processing | OpenCV + OCR |
| Data Validation | Pydantic |
| Visual Verification | Vision-Language Model |
| Frontend | NiceGUI |
| Containerization | Docker / Docker Compose |
| Language | Python |

---

## Limitations

- **Remote sources may be inaccessible** — Quest1 can't process a remote source unreachable from the current network or environment (e.g. OK.ru, protected CDN streams). Downloading the video separately and using local-file mode is the fallback.
- **Spoken dialogue may have no visual representation** — speech recognition can locate a line that OCR can never confirm visually, because it simply never appears on screen.
- **OCR depends on visual quality** — small text, low contrast, stylized fonts, motion, compression, complex backgrounds, and partial occlusion can all degrade OCR performance.
- **Higher tiers require more computation** — dense OCR and VLM verification are more expensive than subtitle matching or speech recognition; the tiered architecture exists to avoid these stages whenever earlier evidence is sufficient.
- **Remote stream formats vary** — URL structure, available tracks, required headers, adaptive-stream formats, and access restrictions differ by provider. Quest1 includes fallback handling for these cases, but ultimate compatibility depends on the source.

---

## Roadmap

- [ ] Add live job-status polling improvements
- [ ] Add pipeline-stage visualization
- [ ] Add candidate-frame visualization
- [ ] Improve VLM candidate arbitration
- [ ] Expand handling of difficult CDN stream formats
- [ ] Improve batch media processing

---

## Repository

https://github.com/josondev/Quest1
