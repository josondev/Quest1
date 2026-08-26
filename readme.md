Quest1: Dynamic Dialogue Detection Engine
Quest1 is an end-to-end media analysis system that locates target spoken or visual dialogue timestamps within local video files and remote streaming URLs (such as OK.ru, CDN streams, and standard MP4 sources). It uses a 5-tier fallback processing pipeline to balance speed, cost, and accuracy.
Technical Highlights & Architecture
1. Robust Media Ingestion (src/ingestion.py)
 * CDN & Adaptive Stream Parsing: Detects non-standard CDN manifests (e.g., path-based tokenized URLs lacking .m3u8 or .mpd extensions) and automatically routes extraction to a full MP4 container cache (allow_download=True).
 * Header Signature Injection: Bypasses CDN hotlink guards by injecting Chrome User-Agent and Referer headers directly into FFmpeg and yt-dlp calls.
 * Audio Codec Verification: Filters out video-only tracks to guarantee valid PCM WAV generation at 16kHz mono.
2. Multi-Tier Fallback Orchestration (src/pipeline.py)
The system attempts the fastest, least compute-intensive tier first before escalating:
 * Tier 0 (Embedded Subtitles): Probes soft subtitles/VTT files and matches targets using sequence similarity.
 * Tier 1 (Audio STT): Demuxes audio directly from streams and aligns word timestamps.
 * Tier 2 (Sparse OCR): Scans visual keyframes across a timeline when audio alignment is low-confidence or missing.
 * Tier 3 (Dense OCR Confirmation): Performs concentrated frame-by-frame text scanning surrounding a detected window to pinpoint exact frame boundaries.
 * Tier 4 (Multi-Candidate VLM Arbitration): Extracts candidate frame windows (C1–C7) around timestamp estimates and routes them to a Vision-Language Model arbiter for spatial text/context validation.
3. Windows OpenCV Guard & Path Routing
 * Remote Stream Protection: Blocks OpenCV (cv2.VideoCapture) from directly opening http(s) streams on Windows platforms to eliminate C++ CAP_IMAGES runtime exceptions, routing all remote frame extractions cleanly through FFmpeg seeking (-ss).
 * Local File Re-routing: When remote streaming fails and triggers the download fallback, video_source automatically updates to the local MP4 disk cache, eliminating unnecessary network calls during downstream frame extraction.
4. Offline Speech-to-Text (src/primary_stt.py)
 * Integrated local faster-whisper (turbo model) running on CPU/GPU via Hugging Face weights, eliminating external cloud API rate limits, payload caps, and network connection drops.
 * Fixed sliding-window word alignment logic to prevent end-timestamp index drift.
5. NiceGUI Control Dashboard (frontend.py)
 * Pure-Python Web Interface built with NiceGUI that connects to the FastAPI backend API.
 * Implements background status polling (/api/v1/jobs/{job_id}) every 2 seconds, showing live progress badges, confidence scores, tier execution tags, and persistent candidate frame rendering.
Project Structure
Quest1/
├── artifacts/                # Generated job outputs, JSON metadata, and persistent frame JPGs
├── src/
│   ├── app.py                # FastAPI REST endpoints & background task router
│   ├── config.py             # System & environment configuration settings
│   ├── fallback_vlm.py       # Vision-Language Model candidate arbitration service
│   ├── ingestion.py          # Stream probing, audio extraction, and FFmpeg wrappers
│   ├── pipeline.py           # 5-tier PipelineOrchestrator implementation
│   ├── primary_ocr.py        # Sparse timeline and dense window OCR scanners
│   ├── primary_stt.py        # Local faster-whisper STT service & alignment logic
│   └── models/
│       └── schemas.py        # Pydantic domain models (JobRequest, DetectionResult, STTResult)
├── frontend.py               # NiceGUI web UI application
└── requirements.txt          # Python dependencies

Quickstart Guide
1. Install Dependencies
Ensure FFmpeg is installed and accessible in your system PATH, then run:
pip install -r requirements.txt
pip install faster-whisper nicegui

2. Launch Backend API Server
Start the FastAPI server on port 8000:
uvicorn src.app:app --reload --port 8000

3. Launch NiceGUI Dashboard
In a separate terminal, launch the web interface:
python frontend.py

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser to submit detection jobs and view visual frame artifacts in real time.
