# Quest1 — Full Deep Dive

Everything you need to explain this project confidently, answer follow-up questions, and understand every design decision.

---

## What Is Quest1?

Quest1 is an **AI-powered video dialogue detection engine**. You give it:
- A video URL (YouTube, CDN stream, local file)
- A line of dialogue (e.g. "in a world where")

It returns:
- The **exact timestamp** (e.g. `00:02:14.320`)
- The **frame number** (e.g. `#3358`)
- A **JPEG screenshot** of that exact moment
- A **confidence score** (e.g. `87.4%`)
- Which **tier** found it (e.g. `Tier 1: STT Acoustic Match`)

---

## Why a Cascade Architecture?

The most important design decision. Instead of running all AI on every job, the pipeline is progressive fallback:

| Tier | Method | Cost | Threshold |
|---|---|---|---|
| 0 | Embedded Subtitle fuzzy match | Free, ms | ≥85% similarity |
| 1 | Whisper STT word alignment | Cheap, seconds | ≥70% confidence |
| 2 | Sparse OCR at 0.5 FPS | Moderate, minutes | ≥60% OCR match |
| 3 | Dense OCR on narrow window | Moderate, seconds | ≥70% OCR match |
| 4 | NVIDIA Llama 3.2 Vision | Expensive | Last resort |

If Tier 0 finds it, nothing else runs. Tier 4 only runs when 0,1,2,3 all fail.

**Interview answer:** "I designed it as a cost-optimised cascade. Running Vision AI on every job would be prohibitively expensive. The system starts with the cheapest signal — subtitles — and escalates only when confidence is insufficient."

---

## Tier 0 — Subtitle Match (ingestion.py)

Downloads VTT subtitle track via yt-dlp. Parses into SubtitleCue objects (start_time, end_time, text). Fuzzy-matches target dialogue against every cue using rapidfuzz.fuzz.ratio(). Accepts if similarity ≥ 85%.

Why first: Pre-computed text metadata. No audio processing, no AI. Milliseconds.
When it fails: No subtitle track, or dialogue is visual-only (burned-in text).

---

## Tier 1 — Speech-to-Text (primary_stt.py)

Extracts audio as WAV via FFmpeg. Sends to Groq Whisper API. Gets back word-level timestamps. Runs sliding window search over all words to find best match.

Chunking: Files ≥10MB are split into 180s segments by FFmpeg. Each chunk transcribed with time offset applied to merge timestamps back.

Confidence formula:
composite = (fuzzy_score × 0.70) + (mean_word_probability × 0.30)

Word probabilities are Whisper's per-word confidence. Low-prob words (mumbled speech) drag the score down, preventing false positives. Accepts if ≥ 0.70.

---

## Tier 3 — Dense OCR (primary_ocr.py)

Triggered by Tier 1 OR Tier 2 as a confirmation step. Extracts every frame at native FPS within a narrow time window. Crops bottom 25% of each frame (subtitle region). Base64-encodes → sends to Mistral OCR. Returns first frame scoring ≥ 70%.

Why crop bottom 25%: Subtitles are always at the bottom. Reduces payload and removes visual noise.

---

## Tier 2 — Sparse OCR (primary_ocr.py)

Scans the entire video but at 0.5 FPS (one frame every 2 seconds). OCRs every sampled frame. Returns best match ≥ 60%. Slow (minutes for long videos) but thorough.

Only runs when Tier 0 and Tier 1 both fail.

---

## Tier 4 — Vision LLM Arbiter (fallback_vlm.py)

Extracts up to 7 candidate frames. Encodes each as base64 JPEG. Sends one at a time to NVIDIA NIM (Llama 3.2 Vision) with prompt: "Is the target dialogue visible in this frame? Respond in JSON."

Parses JSON response for selected_candidate_id, confidence_score, exact_detected_text.
One image per request because NVIDIA NIM's API constraint.

---

## app.py — FastAPI Backend

### The 202 Async Pattern
Pipeline takes up to 2 minutes. Synchronous HTTP would timeout.

POST /api/v1/jobs → generates job_id, stores PROCESSING, queues background task, returns 202 IMMEDIATELY.
Background task runs full pipeline.
Client polls GET /api/v1/jobs/{job_id} every 1.5 seconds.

### Endpoints
POST   /api/v1/jobs              Submit job
GET    /api/v1/jobs/{id}         Poll for result
GET    /api/v1/jobs/{id}/frame   Get JPEG frame
GET    /health                   Container health check
GET    /docs                     Swagger UI (auto-generated)

---

## storage.py — Dual-Mode Storage

JobStore: Azure Table Storage in production, in-memory dict locally.
ArtifactStore: Azure Blob in production, local disk locally.

Identical interface either way — code outside storage.py never knows which backend runs.

Why Azure Table: Container Apps scales to 0 replicas when idle. Fresh container = no in-memory state. Azure Table persists across restarts.

---

## ingestion.py — Stream Layer

probe_metadata(url): yt-dlp extracts fps, duration, subtitle availability, CDN stream URL.
_get_cookies_path(): Downloads cookies.txt from Azure Blob. Validates by comparing file size against blob size — if they differ, re-downloads. Prevents yt-dlp from overwriting auth cookies.
extract_audio_stream(): yt-dlp + FFmpeg → WAV.
extract_frame_on_demand(): FFmpeg -ss [timestamp] -vframes 1 → JPEG.

YouTube auth: tv_embedded player_client + Deno/EJS runtime solves JS challenges.

---

## schemas.py — Pydantic v2 Models

JobRequest: url/local_file_path (one required), target_text (non-empty).
DetectionResult: job_id, status, timestamp_seconds, formatted_timestamp, frame_number, confidence_score, tier_executed, extracted_text, frame_image_path.
JobStatus enum: pending / processing / completed / failed / uncertain / ambiguous.
TierType enum: Tier 0 through Tier 4.
WordTimestamp: word, start, end, probability.

---

## calculate_confidence_score()

score = (0.50 × text_similarity) + (0.30 × ocr_confidence) + (0.20 × alignment_score)

50% text similarity = primary signal
30% OCR confidence = strong secondary evidence
20% alignment = bonus when multiple tiers agree

Degrades gracefully: if OCR not run, ocr_confidence=0.0, other weights still apply.

---

## The YouTube Bot Detection Problem

Cloud IPs (Azure, AWS) are blocked by YouTube unless requests look like a real browser.

Attempts in order:
1. Raw yt-dlp → blocked
2. Webshare free proxies → datacenter IPs also flagged
3. Azure default IP → also flagged
4. IPRoyal free public proxies → dead proxies
5. Android player_client hack → bypasses JS but not auth
6. Deno + yt-dlp EJS → solves JS challenges ✓
7. Fresh cookies + EJS → works ✓

Bugs found along the way:
- yt-dlp overwrites cookies with stripped version → fixed by blob size validation
- Permission denied on temp_data/ → fixed by Dockerfile chown order
- Cookies expire every few weeks → automated via GitHub Actions refresh workflow

---

## Dockerfile Key Decisions

FROM python:3.11-slim: Minimal Debian. ~150MB vs ~900MB for full image.
Copy requirements.txt before code: Docker layer caching — pip re-runs only when deps change.
Deno in /usr/local: System-wide install so non-root appuser can access it.
mkdir before adduser + chown -R: Ensures appuser owns temp_data/ before running.
HEALTHCHECK: Azure Container Apps uses this to know the container is ready.
Single image for both containers: Same Dockerfile, different CMD in Container Apps config.

---

## CI/CD Pipeline

git push origin Rex
  → Job 1: pytest (82 tests) — blocks everything if any fail
  → Job 2: docker build + push to ACR with git SHA tag
  → Job 3: az containerapp update with new SHA
  → Job 4: GitHub Pages deploy from docs/ (parallel with Job 2)

git push with cookies.txt changed
  → refresh-cookies workflow only:
    Upload cookies.txt to Azure Blob
    Bump COOKIES_REFRESH env var → new revision → fresh containers

Why SHA tags not latest: SHA is immutable. Always know exactly what commit is live.

---

## Azure Infrastructure

quest1-rg          Resource Group (Central India)
quest1registry001  ACR — private Docker registry
quest1-env         Container Apps Environment — shared networking
quest1-api         Backend — 0.25 CPU, 0.5Gi RAM, 0-2 replicas, scales to zero
quest1-ui          Frontend — same specs
quest1storage001   Storage Account
  quest1-artifacts  Blob container — frames + cookies.txt
  quest1jobs        Table — job state

Scale to zero: minReplicas=0. When idle, zero containers, zero cost. Cold start ~30s.

---

## Test Suite (82 Tests)

test_ingestion.py   — subtitle parsing, VTT timestamps, stream detection
test_pipeline.py    — full pipeline mock runs, tier logic, confidence scoring
test_primary_stt.py — word alignment, sliding window, chunking
test_primary_ocr.py — frame cropping, OCR response parsing
test_fallback_vlm.py — VLM decision parsing, base64 encoding
test_storage.py     — dual-mode storage, serialisation
test_schemas.py     — Pydantic validation, edge cases

All external APIs (Groq, Mistral, NVIDIA) are mocked — no real calls in CI.

---

## Interview Q&A

Q: Why FastAPI not Flask?
A: FastAPI is async-native. Background tasks, Pydantic v2 integration, auto Swagger docs. Flask would need extra libraries for all three.

Q: How does it handle concurrent jobs?
A: Each pipeline runs in a separate thread from FastAPI's thread pool. Azure Table writes are atomic. Up to 2 replicas handle requests independently.

Q: Why not Celery + Redis?
A: FastAPI BackgroundTasks is sufficient at this scale. Celery + Redis = two more containers, more cost. Would migrate if job volume grew.

Q: What happens if pipeline crashes?
A: The try/except in _execute_pipeline_background catches everything and writes a FAILED DetectionResult. Client sees status: failed with the error message.

Q: Why not YouTube Data API?
A: YouTube Data API gives metadata only. No video stream, audio, or subtitle data. yt-dlp is the only practical tool for actual media access.

Q: How do you handle long videos?
A: STT layer chunks audio into 180s segments via FFmpeg, transcribes each, applies time offset to merge timestamps. Handles arbitrary length.

Q: How would you scale this?
A: 1) Increase maxReplicas. 2) Add Azure Service Bus + Celery workers. 3) Replace Azure Table with Cosmos DB. 4) Redis cache for subtitle downloads.

Q: Why cookies in Azure Blob not baked into Docker image?
A: Cookies expire every few weeks. Baking them into the image would require a full rebuild on every rotation. Blob storage allows hot-update without redeployment.
