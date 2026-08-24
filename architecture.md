# System Architecture: Hybrid Video Dialogue Frame Detection

## 1. Executive Summary & Problem Formulation

The objective of this system is to ingest a video URL (e.g., YouTube, OK.ru) and a target dialogue string (e.g., *"My mind rebels at stagnation"*), then autonomously identify and extract:
1. **The Exact Timestamp** ($T_{\text{onset}}$ in `HH:MM:SS.sss` format) where the dialogue first appears.
2. **The Exact Frame Number** ($\text{Frame} = \text{round}(T \times \text{FPS})$).
3. **The Extracted Dialogue Text** verified via multi-modal alignment.
4. **The Verified Frame Image & Cropped Dialogue ROI** saved as visual artifacts.

---

## 2. End-to-End Architectural Flow

```
                      +------------------------------------------+
                      |   User Input (Video URL + Target Text)   |
                      +--------------------+---------------------+
                                           |
                      +--------------------v---------------------+
                      |       Direct Stream Ingestion            |
                      |   yt-dlp: Probe metadata (FPS, dur)      |
                      |   Extract lightweight audio stream only  |
                      +--------------------+---------------------+
                                           |
                      +--------------------v---------------------+
                      | Tier 0: Embedded Subtitle Check          |
                      | Check .vtt / .srt / .ass container subs  |
                      +----------+--------------------+----------+
                                 |                    |
                         Found Subtitle?        No Subs / Miss
                                 |                    |
                                 |          +---------v----------+
                                 |          | Tier 1: Audio STT  |
                                 |          | faster-whisper     |
                                 |          | Word timestamps    |
                                 |          +----+----------+----+
                                 |               |          |
                                 |          Found Match?    No Speech / Music
                                 |               |          |
                                 |               |    +-----v----------------------+
                                 |               |    | Tier 2: Sparse Timeline OCR|
                                 |               |    | Mistral OCR (0.5 - 1 FPS)  |
                                 |               |    +-----+----------------------+
                                 |               |          |
                                 +-------+-------+----------+
                                         |
                       Promising Candidate Window [t_start - 2s, t_end + 2s]
                                         |
                      +------------------v-----------------------+
                      | Tier 3: Dense Native FPS Sampling        |
                      | Slice ~60-90 frames at native video FPS  |
                      | Subtitle ROI Crop & Mistral OCR Onset    |
                      +------------------+-----------------------+
                                         |
                      +------------------v-----------------------+
                      | Confidence Fusion Engine                 |
                      | Score = w_sim*S + w_ocr*C + w_align*A    |
                      +----------+--------------------+----------+
                                 |                    |
                         Score >= 0.85          Score < 0.85 / Ambiguous
                                 |                    |
                                 |          +---------v--------------------+
                                 |          | Tier 4: Fallback VLM Arbiter |
                                 |          | Open-Source Qwen2.5-VL       |
                                 |          | 5-8 Candidate Frame Cluster  |
                                 |          +---------+--------------------+
                                 |                    |
                      +----------v--------------------v----------+
                      | Mathematical Grounding & Artifact Generation |
                      | Frame Number = round(Timestamp * FPS)    |
                      | Crop Dialogue Bounding Box & Save JPG    |
                      +------------------+-----------------------+
                                         |
                      +------------------v-----------------------+
                      | Presentation Layer                       |
                      | 1. FastAPI REST Endpoint (/find-frame)   |
                      | 2. Reflex Reactive Web Interface         |
                      +------------------------------------------+
```

---

## 3. Tier-by-Tier Technical Breakdown

### Ingestion: Zero-Bloat Stream Handling
- **Mechanism**: `yt-dlp` probes container metadata (duration, nominal FPS, codec, resolution, available subtitle tracks) without downloading video streams.
- **Audio Extraction**: Streams only lightweight compressed audio (`opus`/`m4a`/`mp3`, ~1-2 MB per minute) directly to memory or a temporary local audio chunk.
- **Security**: Treats URLs as untrusted input, strictly enforcing `http`/`https` schemes and preventing server-side request forgery (SSRF).

---

### Tier 0: Zero-Cost Embedded Subtitle Probing
- **Inspection**: Probes for soft subtitle streams (`.vtt`, `.srt`, `.ass`, CEA-608/708).
- **Matching**: Parses time intervals and tests for text similarity.
- **Benefit**: Runs in $<100\,\text{ms}$ with zero GPU/AI inference cost. If a match is found, jumps directly to Tier 3 for dense frame onset extraction.

---

### Tier 1: Primary Acoustic Path (Whisper STT)
- **Engine**: `faster-whisper` (CTranslate2 optimized local Whisper).
- **Execution**: Computes word-level timestamps (`word_timestamps=True`).
- **Phrase Alignment**: Uses token-sort and normalized Levenshtein string matching against normalized target dialogue:
  $$\text{Similarity}(S_1, S_2) = 1 - \frac{\text{Levenshtein}(S_1, S_2)}{\max(|S_1|, |S_2|)}$$
- **Candidate Window**: Emits $[t_{\text{start}} - 2.0\,\text{s}, t_{\text{end}} + 2.0\,\text{s}]$ to account for typical subtitle lead/lag times.

---

### Tier 2: Visual Safety Net (Sparse Timeline OCR)
- **Trigger**: Activated if Tier 0 and Tier 1 miss (e.g., silent film, speech drowned by loud music, foreign audio with English subtitles, or text-only intertitle).
- **Sampling**: Slices video sparsely at $0.5 - 1.0\,\text{FPS}$ using FFmpeg stream seeking.
- **Engine**: **Mistral OCR** (`mistral-ocr-latest`) extracts text from subtitle regions across sampled frames.
- **Output**: Identifies the highest-likelihood visual timestamp window.

---

### Tier 3: Dense Native FPS Rescan & Exact Onset Detection
- **Sampling**: Extracts frames at full native video FPS across the candidate window ($\approx 60 - 90$ frames total).
- **ROI Optimization**: Focuses OCR on the lower third of the video frame (subtitle zone) with full-frame fallback.
- **Onset Discovery**: Iterates frame-by-frame to find the **earliest continuous frame** where the dialogue appears, avoiding midpoint or fade-out inaccuracies.

---

### Tier 4: Fallback Open-Source VLM Arbiter
- **Trigger**: Activated when composite confidence score is $< 0.85$, text is heavily stylized/cursive, or STT and OCR temporal locations disagree.
- **Engine**: Open-Source Vision-Language Model (**`Qwen2.5-VL`** / **`Llama-3.2-Vision`** / **`SmolVLM`**) via local Ollama or HuggingFace Transformers.
- **Candidate Cluster Protocol**:
  - Feeds a bounded cluster of 5–8 candidate frames: `[C1, C2, C3, ...]`.
  - Prompts with strict JSON schema requiring:
    ```json
    {
      "selected_candidate_id": "C3",
      "exact_detected_text": "My mind rebels at stagnation",
      "bounding_box": [ymin, xmin, ymax, xmax],
      "confidence_score": 0.95,
      "reasoning": "Candidate C3 is the first frame where the full dialogue is visually legible."
    }
    ```
  - **Zero Hallucination Guarantee**: The VLM selects from physical candidate frames; timestamps are resolved deterministically from container metadata.

---

## 4. Mathematical Grounding & Frame Calculations

### Constant Frame Rate (CFR) Formula:
$$\text{Frame Index} = \text{round}(T_{\text{seconds}} \times \text{FPS})$$

### Formatted Timestamp:
$$T = \lfloor H \rfloor : \lfloor M \rfloor : \lfloor S \rfloor . \text{millis}$$

### Variable Frame Rate (VFR) Edge Cases:
For streams with variable frame rates or non-integer timebases, the system uses FFmpeg presentation timestamps (PTS) and sequential frame decoding (`select='eq(n\,...)'`) to ensure the exact matching physical frame is saved.

---

## 5. Confidence Fusion Engine

The composite confidence score $C_{\text{final}}$ is calculated as:
$$C_{\text{final}} = w_{\text{sim}} \cdot S_{\text{text}} + w_{\text{ocr}} \cdot C_{\text{ocr}} + w_{\text{align}} \cdot A_{\text{temporal}}$$

- $S_{\text{text}} \in [0, 1]$: Fuzzy text similarity between target dialogue and extracted text.
- $C_{\text{ocr}} \in [0, 1]$: Model-reported recognition confidence.
- $A_{\text{temporal}} \in [0, 1]$: Temporal alignment agreement between audio STT and visual OCR.
- **Threshold**:
  - $C_{\text{final}} \ge 0.85$: Accepted deterministically.
  - $C_{\text{final}} < 0.85$: Routed to Tier 4 VLM Arbiter.

---

## 6. Presentation & Serving Layer

1. **FastAPI Backend**:
   - `POST /api/v1/find-frame`: Asynchronous job execution accepting URL and target dialogue.
   - `GET /api/v1/jobs/{job_id}`: Real-time status, progress tracking, and final payload.
   - `GET /api/v1/artifacts/{artifact_id}`: Serves cropped dialogue and full-frame JPGs.
   - Auto-generated OpenAPI documentation at `/docs`.

2. **Reflex Reactive Frontend**:
   - Built 100% in Python using Reflex.
   - Interactive URL and target phrase submission.
   - Live execution stage progress indicator.
   - Video player synced to the detected timestamp.
   - Side-by-side display of full video frame and cropped bounding box dialogue.

---

## 7. Technical Interview Defense Summary

| Question | Defensible Engineering Answer |
| :--- | :--- |
| **Why not run OCR across the entire video?** | A 60-minute 1080p video at 30 FPS has **108,000 frames**. Running OCR on all frames is computationally wasteful. STT narrows the search to **~60 frames** in 2 seconds. |
| **Why use Mistral OCR?** | Avoids complex C++ binaries and OS-specific compilation issues (Tesseract / PaddlePaddle DLLs) while providing state-of-the-art visual text recognition. |
| **Why use an Open-Source VLM?** | Ensures 100% local privacy, zero proprietary API cost, and high-performance multi-frame visual reasoning via `Qwen2.5-VL` or `Llama-3.2-Vision`. |
| **How do you guarantee the VLM doesn't hallucinate timestamps?** | The VLM never generates timestamps. It only selects a candidate ID from a bounded cluster of physical frames extracted by FFmpeg. |
