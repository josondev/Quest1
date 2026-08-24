"""
src/primary_ocr.py
Step 4: Visual Frame Sampling + Mistral OCR

Implements Tier 2 (sparse timeline scan) and Tier 3 (dense window onset detection)
using OpenCV for frame extraction and Mistral OCR (mistral-ocr-latest) for visual
text extraction.

Mistral OCR API reference: https://docs.mistral.ai/api/endpoint/ocr
SDK: mistralai v2.9.4  →  from mistralai.client.sdk import Mistral
"""

import base64
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rapidfuzz import fuzz

from mistralai.client.sdk import Mistral

from src.models.schemas import BoundingBox, CandidateFrame, OCRResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class OCRError(Exception):
    """Raised when a Mistral OCR call fails or returns no usable content."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def frame_to_base64(frame: np.ndarray) -> str:
    """
    Encode an OpenCV BGR frame as a JPEG base64 string.

    The Mistral OCR API accepts images as:
        "data:image/jpeg;base64,<base64_string>"
    Ref: https://docs.mistral.ai/api/endpoint/ocr
    """
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise OCRError("Failed to encode video frame as JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def crop_subtitle_roi(frame: np.ndarray, roi_fraction: float = 0.25) -> np.ndarray:
    """
    Crop the bottom portion of a frame to focus OCR on the subtitle zone.

    Subtitles in broadcast/film content appear in the bottom 20-25% of the frame.
    Reducing the OCR region minimises token usage and improves text isolation.

    Args:
        frame: Full-resolution OpenCV BGR frame.
        roi_fraction: Fraction of frame height to retain from the bottom (default 0.25).

    Returns:
        Cropped frame.  Falls back to full frame if the crop would be < 32px tall.
    """
    h, w = frame.shape[:2]
    crop_start = int(h * (1.0 - roi_fraction))
    cropped = frame[crop_start:h, 0:w]
    if cropped.shape[0] < 32:
        logger.debug("ROI crop too small (%dpx), using full frame.", cropped.shape[0])
        return frame
    return cropped


def _extract_text_from_response(response) -> Tuple[str, float, Optional[BoundingBox]]:
    """
    Parse a Mistral OCR response object into (full_text, confidence, bounding_box).

    The OCR response structure (mistral-ocr-latest):
        response.pages[0].markdown         – full extracted markdown text
        response.pages[0].blocks[i]        – per-block detail (when include_blocks=True)
            .content                       – block text
            .top_left_x / .top_left_y / .bottom_right_x / .bottom_right_y
    """
    if not response.pages:
        return "", 0.0, None

    page = response.pages[0]
    full_text: str = page.markdown or ""

    # Use the first block's geometry as the primary bounding box
    primary_box: Optional[BoundingBox] = None
    if hasattr(page, "blocks") and page.blocks:
        try:
            block = page.blocks[0]
            primary_box = BoundingBox(
                ymin=float(getattr(block, "top_left_y", 0.0)),
                xmin=float(getattr(block, "top_left_x", 0.0)),
                ymax=float(getattr(block, "bottom_right_y", 1.0)),
                xmax=float(getattr(block, "bottom_right_x", 1.0)),
            )
        except Exception:
            pass

    # Mistral OCR does not expose per-block confidence in all SDK versions.
    # Estimate confidence from text richness: longer text = higher confidence.
    confidence = min(1.0, len(full_text.strip()) / 50.0) if full_text.strip() else 0.0

    return full_text.strip(), confidence, primary_box


# ---------------------------------------------------------------------------
# Core OCR call
# ---------------------------------------------------------------------------

def ocr_frame(
    frame: np.ndarray,
    client: Mistral,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    use_roi: bool = True,
) -> OCRResult:
    """
    Send a single video frame to Mistral OCR and return a structured result.

    Uses type "image_url" with a base64 data URI per the official API spec:
        document = {"type": "image_url", "image_url": "data:image/jpeg;base64,..."}

    Args:
        frame:             OpenCV BGR frame (full resolution or pre-cropped).
        client:            An authenticated Mistral SDK client instance.
        frame_index:       Frame number within the video (for schema compliance).
        timestamp_seconds: Timestamp of this frame in seconds.
        use_roi:           If True, crops the bottom 25% (subtitle zone) before sending.

    Returns:
        OCRResult with extracted text, bounding box, and confidence.
    """
    target_frame = crop_subtitle_roi(frame) if use_roi else frame
    b64 = frame_to_base64(target_frame)

    try:
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64}",
            },
            include_blocks=True,
        )
    except Exception as exc:
        raise OCRError(f"Mistral OCR API call failed: {exc}") from exc

    text, confidence, box = _extract_text_from_response(response)

    return OCRResult(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        detected_text=text,
        confidence=confidence,
        bounding_box=box,
    )


# ---------------------------------------------------------------------------
# Frame sampling utilities
# ---------------------------------------------------------------------------

def sample_frames_sparse(
    video_path: Path,
    sample_fps: float = 0.5,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Extract frames at a reduced rate (default 0.5 FPS = 1 frame every 2 seconds).

    Used in Tier 2: sparse timeline scan across the full video duration.

    Args:
        video_path:  Path to the downloaded video file.
        sample_fps:  Frames per second to sample (0.5 → 1 frame every 2 s).

    Returns:
        List of (frame_number, timestamp_seconds, frame_bgr_image).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OCRError(f"Cannot open video file: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval: int = max(1, int(round(native_fps / sample_fps)))

    sampled: List[Tuple[int, float, np.ndarray]] = []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / native_fps
        sampled.append((frame_idx, round(timestamp, 3), frame))
        frame_idx += frame_interval

    cap.release()
    logger.info(
        "Sparse sampling: %d frames extracted at %.2f FPS from %s",
        len(sampled), sample_fps, video_path.name,
    )
    return sampled


def sample_frames_dense(
    video_path: Path,
    t_start: float,
    t_end: float,
    native_fps: float,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Extract every frame between t_start and t_end at the native video FPS.

    Used in Tier 3: dense window scan around the candidate timestamp found
    by either audio STT (Tier 1) or sparse OCR (Tier 2).

    Args:
        video_path:  Path to the downloaded video file.
        t_start:     Window start in seconds.
        t_end:       Window end in seconds.
        native_fps:  Video's native frame rate (from probe_metadata).

    Returns:
        List of (frame_number, timestamp_seconds, frame_bgr_image).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OCRError(f"Cannot open video file: {video_path}")

    start_frame = max(0, int(t_start * native_fps))
    end_frame = int(t_end * native_fps)

    sampled: List[Tuple[int, float, np.ndarray]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / native_fps
        sampled.append((frame_idx, round(timestamp, 3), frame))

    cap.release()
    logger.info(
        "Dense sampling: %d frames in window [%.2fs, %.2fs] from %s",
        len(sampled), t_start, t_end, video_path.name,
    )
    return sampled


# ---------------------------------------------------------------------------
# Tier 2: Sparse Timeline Scan
# ---------------------------------------------------------------------------

def scan_sparse_timeline(
    video_path: Path,
    target_dialogue: str,
    client: Mistral,
    sample_fps: float = 0.5,
    match_threshold: float = 60.0,
) -> Optional[CandidateFrame]:
    """
    Tier 2 — Sparse OCR scan across the full video timeline.

    Samples frames at 0.5 FPS, runs Mistral OCR on the subtitle ROI of each frame,
    and returns the highest-matching CandidateFrame or None if no match is found.

    Args:
        video_path:        Path to the downloaded video file.
        target_dialogue:   The target dialogue text to search for.
        client:            Authenticated Mistral SDK client.
        sample_fps:        Sampling rate (default 0.5 FPS).
        match_threshold:   Minimum fuzzy match score (0-100) to consider a hit.

    Returns:
        CandidateFrame for the best-matching frame, or None.
    """
    normalized_target = target_dialogue.lower().strip()
    frames = sample_frames_sparse(video_path, sample_fps=sample_fps)

    best_candidate: Optional[CandidateFrame] = None
    best_score: float = 0.0

    for frame_number, timestamp, frame in frames:
        try:
            result = ocr_frame(frame, client, frame_index=frame_number, timestamp_seconds=timestamp, use_roi=True)
        except OCRError as exc:
            logger.warning("OCR failed for frame %d (%.2fs): %s", frame_number, timestamp, exc)
            continue

        if not result.detected_text:
            continue

        score = fuzz.partial_ratio(normalized_target, result.detected_text.lower())
        logger.debug(
            "Frame %d | %.2fs | score %.1f | '%s'",
            frame_number, timestamp, score, result.detected_text[:80],
        )

        if score > best_score and score >= match_threshold:
            best_score = score
            best_candidate = CandidateFrame(
                candidate_id=f"C{frame_number}",
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                image_path="",  # populated downstream by pipeline
                ocr_detected_text=result.detected_text,
                ocr_confidence=result.confidence,
            )

    if best_candidate:
        logger.info(
            "Tier 2 sparse scan found candidate at frame %d (%.2fs) with score %.1f",
            best_candidate.frame_number, best_candidate.timestamp_seconds, best_score,
        )
    else:
        logger.info("Tier 2 sparse scan: no match found above threshold %.1f.", match_threshold)

    return best_candidate


# ---------------------------------------------------------------------------
# Tier 3: Dense Window Onset Detection
# ---------------------------------------------------------------------------

def scan_dense_window(
    video_path: Path,
    t_start: float,
    t_end: float,
    native_fps: float,
    target_dialogue: str,
    client: Mistral,
    match_threshold: float = 70.0,
) -> Optional[CandidateFrame]:
    """
    Tier 3 — Dense OCR scan within a narrow candidate time window.

    Runs OCR on every frame in [t_start, t_end] and returns the EARLIEST
    frame where the target dialogue first appears (onset detection).

    Args:
        video_path:      Path to the downloaded video file.
        t_start:         Window start in seconds.
        t_end:           Window end in seconds.
        native_fps:      Native FPS from VideoMetadata.
        target_dialogue: Target text to find.
        client:          Authenticated Mistral SDK client.
        match_threshold: Minimum fuzzy score to accept as a match.

    Returns:
        CandidateFrame for the FIRST (onset) frame where dialogue appears, or None.
    """
    normalized_target = target_dialogue.lower().strip()
    frames = sample_frames_dense(video_path, t_start, t_end, native_fps)

    for frame_number, timestamp, frame in frames:
        try:
            result = ocr_frame(frame, client, frame_index=frame_number, timestamp_seconds=timestamp, use_roi=True)
        except OCRError as exc:
            logger.warning("OCR failed for frame %d (%.2fs): %s", frame_number, timestamp, exc)
            continue

        if not result.detected_text:
            continue

        score = fuzz.partial_ratio(normalized_target, result.detected_text.lower())
        logger.debug(
            "Dense frame %d | %.2fs | score %.1f | '%s'",
            frame_number, timestamp, score, result.detected_text[:80],
        )

        if score >= match_threshold:
            logger.info(
                "Tier 3 dense scan: onset detected at frame %d (%.2fs) with score %.1f",
                frame_number, timestamp, score,
            )
            return CandidateFrame(
                candidate_id=f"C{frame_number}",
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                image_path="",  # populated downstream by pipeline
                ocr_detected_text=result.detected_text,
                ocr_confidence=result.confidence,
            )

    logger.info("Tier 3 dense scan: no onset found in [%.2fs, %.2fs].", t_start, t_end)
    return None
