import base64
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from rapidfuzz import fuzz
from mistralai.client import Mistral

from src.models.schemas import BoundingBox, CandidateFrame, OCRResult

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Raised when a Mistral OCR call fails or returns no usable content."""
    pass


def frame_to_base64(frame: np.ndarray) -> str:
    """Encode an OpenCV BGR frame as a JPEG base64 string."""
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise OCRError("Failed to encode video frame as JPEG.")

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


def crop_subtitle_roi(
    frame: np.ndarray,
    roi_fraction: float = 0.25,
) -> np.ndarray:
    """Crop the bottom portion of a frame to focus OCR on subtitles."""
    h, w = frame.shape[:2]

    crop_start = int(
        h * (1.0 - roi_fraction)
    )

    cropped = frame[
        crop_start:h,
        0:w,
    ]

    if cropped.shape[0] < 32:
        logger.debug(
            "ROI crop too small (%dpx), using full frame.",
            cropped.shape[0],
        )
        return frame

    return cropped


def _extract_text_from_response(
    response,
) -> Tuple[str, float, Optional[BoundingBox]]:
    """Parse a Mistral OCR response into text, confidence, and bounding box."""
    if not hasattr(response, "pages") or not response.pages:
        return "", 0.0, None

    page = response.pages[0]

    full_text: str = (
        getattr(page, "markdown", "") or ""
    )

    primary_box: Optional[BoundingBox] = None

    if hasattr(page, "blocks") and page.blocks:
        try:
            block = page.blocks[0]

            primary_box = BoundingBox(
                ymin=float(
                    getattr(
                        block,
                        "top_left_y",
                        0.0,
                    )
                ),
                xmin=float(
                    getattr(
                        block,
                        "top_left_x",
                        0.0,
                    )
                ),
                ymax=float(
                    getattr(
                        block,
                        "bottom_right_y",
                        1.0,
                    )
                ),
                xmax=float(
                    getattr(
                        block,
                        "bottom_right_x",
                        1.0,
                    )
                ),
            )
        except Exception:
            primary_box = None

    confidence = (
        1.0
        if full_text.strip()
        else 0.0
    )

    return (
        full_text.strip(),
        confidence,
        primary_box,
    )


def ocr_frame(
    frame: np.ndarray,
    client: Mistral,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    use_roi: bool = True,
) -> OCRResult:
    """Send one frame to Mistral OCR and return a structured result."""
    target_frame = (
        crop_subtitle_roi(frame)
        if use_roi
        else frame
    )

    b64 = frame_to_base64(target_frame)

    try:
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": (
                    f"data:image/jpeg;base64,{b64}"
                ),
            },
            include_blocks=True,
        )

    except Exception as exc:
        raise OCRError(
            f"Mistral OCR API call failed: {exc}"
        ) from exc

    text, confidence, box = (
        _extract_text_from_response(response)
    )

    return OCRResult(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        detected_text=text,
        confidence=confidence,
        bounding_box=box,
    )


def sample_frames_sparse(
    video_path: Union[Path, str],
    sample_fps: float = 0.5,
) -> List[Tuple[int, float, np.ndarray]]:
    """Extract frames at a reduced sampling rate."""
    if sample_fps <= 0:
        raise ValueError(
            "sample_fps must be greater than 0."
        )

    source_str = str(video_path)

    cap = cv2.VideoCapture(
        source_str
    )

    if not cap.isOpened():
        raise OCRError(
            f"Cannot open video file: {source_str}"
        )

    try:
        native_fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or 25.0
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frame_interval = max(
            1,
            int(
                round(
                    native_fps / sample_fps
                )
            ),
        )

        sampled: List[
            Tuple[int, float, np.ndarray]
        ] = []

        frame_idx = 0

        while (
            frame_idx < total_frames
            or total_frames <= 0
        ):
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            timestamp = (
                frame_idx / native_fps
            )

            sampled.append(
                (
                    frame_idx,
                    round(timestamp, 3),
                    frame,
                )
            )

            for _ in range(
                frame_interval - 1
            ):
                if not cap.grab():
                    break

                frame_idx += 1

            frame_idx += 1

        logger.info(
            "Sparse sampling: %d frames extracted from %s",
            len(sampled),
            source_str,
        )

        return sampled

    finally:
        cap.release()


def sample_frames_dense(
    video_path: Union[Path, str],
    t_start: float,
    t_end: float,
    native_fps: float,
) -> List[Tuple[int, float, np.ndarray]]:
    """Extract every frame between t_start and t_end at native FPS."""
    if native_fps <= 0:
        raise ValueError(
            "native_fps must be greater than 0."
        )

    if t_end < t_start:
        raise ValueError(
            "t_end must be greater than or equal to t_start."
        )

    source_str = str(video_path)

    cap = cv2.VideoCapture(
        source_str
    )

    if not cap.isOpened():
        raise OCRError(
            f"Cannot open video file: {source_str}"
        )

    try:
        start_frame = max(
            0,
            int(t_start * native_fps),
        )

        end_frame = max(
            start_frame,
            int(t_end * native_fps),
        )

        sampled: List[
            Tuple[int, float, np.ndarray]
        ] = []

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            float(start_frame),
        )

        for frame_idx in range(
            start_frame,
            end_frame + 1,
        ):
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            timestamp = (
                frame_idx / native_fps
            )

            sampled.append(
                (
                    frame_idx,
                    round(timestamp, 3),
                    frame,
                )
            )

        logger.info(
            "Dense sampling: %d frames extracted from %s",
            len(sampled),
            source_str,
        )

        return sampled

    finally:
        cap.release()


def scan_sparse_timeline(
    video_path: Union[Path, str],
    target_dialogue: str,
    client: Mistral,
    sample_fps: float = 0.5,
    match_threshold: float = 60.0,
    save_frame_path: Optional[Path] = None,
) -> Optional[CandidateFrame]:
    """Tier 2 — Sparse OCR scan across the full video timeline."""
    normalized_target = (
        target_dialogue.lower().strip()
    )

    frames = sample_frames_sparse(
        video_path,
        sample_fps=sample_fps,
    )

    best_candidate: Optional[
        CandidateFrame
    ] = None

    best_score = 0.0
    best_frame: Optional[np.ndarray] = None

    for (
        frame_number,
        timestamp,
        frame,
    ) in frames:
        try:
            result = ocr_frame(
                frame,
                client,
                frame_index=frame_number,
                timestamp_seconds=timestamp,
                use_roi=True,
            )

        except OCRError as exc:
            logger.warning(
                "OCR failed for frame %d (%.2fs): %s",
                frame_number,
                timestamp,
                exc,
            )
            continue

        if not result.detected_text:
            continue

        score = fuzz.partial_ratio(
            normalized_target,
            result.detected_text.lower(),
        )

        if (
            score > best_score
            and score >= match_threshold
        ):
            best_score = score
            best_frame = frame

            best_candidate = CandidateFrame(
                candidate_id=f"C{frame_number}",
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                image_path="",
                ocr_detected_text=result.detected_text,
                ocr_confidence=round(
                    score / 100.0,
                    4,
                ),
            )

    if (
        best_candidate is not None
        and best_frame is not None
        and save_frame_path is not None
    ):
        save_frame_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if cv2.imwrite(
            str(save_frame_path),
            best_frame,
        ):
            best_candidate.image_path = str(
                save_frame_path
            )
        else:
            logger.warning(
                "Failed to persist Tier 2 matched frame to %s",
                save_frame_path,
            )

    return best_candidate


def scan_dense_window(
    video_path: Union[Path, str],
    t_start: float,
    t_end: float,
    native_fps: float,
    target_dialogue: str,
    client: Mistral,
    match_threshold: float = 70.0,
    save_frame_path: Optional[Path] = None,
) -> Optional[CandidateFrame]:
    """Tier 3 — Dense OCR onset detection within a candidate window."""
    normalized_target = (
        target_dialogue.lower().strip()
    )

    frames = sample_frames_dense(
        video_path,
        t_start,
        t_end,
        native_fps,
    )

    for (
        frame_number,
        timestamp,
        frame,
    ) in frames:
        try:
            result = ocr_frame(
                frame,
                client,
                frame_index=frame_number,
                timestamp_seconds=timestamp,
                use_roi=True,
            )

        except OCRError as exc:
            logger.warning(
                "OCR failed for frame %d (%.2fs): %s",
                frame_number,
                timestamp,
                exc,
            )
            continue

        if not result.detected_text:
            continue

        score = fuzz.partial_ratio(
            normalized_target,
            result.detected_text.lower(),
        )

        if score < match_threshold:
            continue

        candidate = CandidateFrame(
            candidate_id=f"C{frame_number}",
            frame_number=frame_number,
            timestamp_seconds=timestamp,
            image_path="",
            ocr_detected_text=result.detected_text,
            ocr_confidence=round(
                score / 100.0,
                4,
            ),
        )

        if save_frame_path is not None:
            save_frame_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if cv2.imwrite(
                str(save_frame_path),
                frame,
            ):
                candidate.image_path = str(
                    save_frame_path
                )
            else:
                logger.warning(
                    "Failed to persist Tier 3 matched frame to %s",
                    save_frame_path,
                )

        return candidate

    return None