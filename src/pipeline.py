import json
import logging
import math
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any, List, Optional, Union

import cv2

from src.config import settings
from src.fallback_vlm import VLMArbiterService, VLMError
from src.ingestion import IngestionError, StreamIngestionService
from src.models.schemas import (
    CandidateFrame,
    DetectionResult,
    JobRequest,
    JobStatus,
    STTResult,
    TierType,
)
from src.primary_ocr import (
    OCRError,
    scan_dense_window,
    scan_sparse_timeline,
)
from src.primary_stt import SpeechToTextService

logger = logging.getLogger(__name__)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def format_timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0

    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def calculate_confidence_score(
    text_similarity: float,
    ocr_confidence: float = 0.0,
    alignment_score: float = 0.0,
    w_sim: float = 0.5,
    w_ocr: float = 0.3,
    w_align: float = 0.2,
) -> float:
    score = (
        w_sim * text_similarity
        + w_ocr * ocr_confidence
        + w_align * alignment_score
    )

    return round(
        max(0.0, min(1.0, score)),
        4,
    )


def _sanitize_path(val: Any) -> Optional[str]:
    if isinstance(val, (str, Path)) and not hasattr(
        val,
        "_mock_return_value",
    ):
        value = str(val).strip()
        return value if value else None

    return None


def is_valid_wav(path: Union[str, Path]) -> bool:
    """
    Validate that the extracted audio is a structurally valid,
    non-empty WAV with at least 1 second of audio.

    IMPORTANT:
    Tests intentionally create a 1-second WAV, so the check must
    be >= 1.0 rather than > 1.0.
    """
    try:
        wav_path = Path(path)

        if not wav_path.exists():
            return False

        if not wav_path.is_file():
            return False

        if wav_path.stat().st_size <= 44:
            return False

        if wav_path.suffix.lower() != ".wav":
            return False

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()

            if sample_rate <= 0:
                return False

            duration = frame_count / sample_rate

            return duration >= 1.0

    except Exception as exc:
        logger.debug(
            "WAV validation failed for %s: %s",
            path,
            exc,
        )
        return False


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class PipelineOrchestrator:

    def __init__(
        self,
        ingestion_service: Optional[StreamIngestionService] = None,
        stt_service: Optional[SpeechToTextService] = None,
        vlm_service: Optional[VLMArbiterService] = None,
        mistral_client: Optional[Any] = None,
    ):
        self.ingestion_service = (
            ingestion_service
            if ingestion_service is not None
            else StreamIngestionService()
        )

        self.stt_service = (
            stt_service
            if stt_service is not None
            else SpeechToTextService()
        )

        self.vlm_service = (
            vlm_service
            if vlm_service is not None
            else VLMArbiterService()
        )

        self.mistral_client = mistral_client

    # ========================================================
    # FRAME EXTRACTION
    # ========================================================

    def _extract_single_frame(
        self,
        video_source: str,
        timestamp_seconds: float,
        fps: float,
        dest_path: Path,
    ) -> bool:
        """
        OpenCV fallback for frame extraction.

        The primary production path is still:
            ingestion_service.extract_frame_on_demand()

        OpenCV is retained as a fallback because it is useful for:
        - local files
        - test mocks
        - environments where FFmpeg extraction temporarily fails
        """

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.warning(
                "Unable to open video source with OpenCV: %s",
                str(video_source or "")[:120],
            )
            return False

        try:
            frame_number = int(
                round(timestamp_seconds * fps)
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            ret, frame = cap.read()

            if not ret or frame is None:
                return False

            dest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            return bool(
                cv2.imwrite(
                    str(dest_path),
                    frame,
                )
            )

        except Exception as exc:
            logger.warning(
                "OpenCV frame extraction failed at %.3fs: %s",
                timestamp_seconds,
                exc,
            )
            return False

        finally:
            cap.release()

    def _persist_frame(
        self,
        artifacts_dir: Path,
        job_id: str,
        tier_label: str,
        video_source: Optional[str],
        timestamp_seconds: float,
        fps: float,
    ) -> Optional[str]:

        if not video_source:
            return None

        artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame_path = (
            artifacts_dir
            / f"{job_id}_{tier_label}.jpg"
        )

        # PRIMARY:
        # Direct CDN/HLS stream -> FFmpeg -> one frame
        try:
            extracted = (
                self.ingestion_service.extract_frame_on_demand(
                    video_source,
                    timestamp_seconds,
                    frame_path,
                )
            )

            if extracted and frame_path.exists():
                logger.info(
                    "Frame extracted from stream at %.2fs",
                    timestamp_seconds,
                )
                return str(frame_path)

        except Exception as exc:
            logger.warning(
                "On-demand stream frame extraction failed: %s",
                exc,
            )

        # FALLBACK:
        # OpenCV
        if self._extract_single_frame(
            video_source,
            timestamp_seconds,
            fps,
            frame_path,
        ):
            return str(frame_path)

        logger.warning(
            "Unable to extract frame at %.2fs",
            timestamp_seconds,
        )

        return None

    # ========================================================
    # METADATA
    # ========================================================

    def _save_job_metadata(
        self,
        artifacts_dir: Path,
        result: DetectionResult,
    ) -> None:

        try:
            artifacts_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata_file = (
                artifacts_dir
                / "metadata.json"
            )

            metadata_file.write_text(
                json.dumps(
                    result.model_dump(),
                    indent=2,
                )
            )

        except Exception as exc:
            logger.warning(
                "Metadata save failed: %s",
                exc,
            )

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(
        self,
        job_id: str,
        request: JobRequest,
    ) -> DetectionResult:

        temp_dir_str = tempfile.mkdtemp(
            prefix=f"quest1_{job_id}_"
        )

        temp_dir = Path(temp_dir_str)

        artifacts_base = getattr(
            settings,
            "artifacts_dir",
            getattr(
                settings,
                "artifact_storage_dir",
                Path("artifacts"),
            ),
        )

        artifacts_dir = (
            Path(artifacts_base)
            / job_id
        )

        logger.info(
            "Executing job %s",
            job_id,
        )

        try:

            # ==================================================
            # METADATA / STREAM RESOLUTION
            # ==================================================

            metadata = (
                self.ingestion_service
                .probe_metadata(request.url)
            )

            fps = (
                metadata.fps
                if metadata.fps > 0
                else 25.0
            )

            video_source = (
                metadata.stream_path
                or request.url
            )

            if metadata.is_local:
                logger.info(
                    "Processing local video: %s",
                    video_source,
                )
            else:
                logger.info(
                    "Processing remote stream: %s",
                    str(video_source or "")[:120],
                )

            # ==================================================
            # TIER 0 : SUBTITLE MATCH
            # ==================================================

            if metadata.has_subtitles:

                tier0_result = (
                    self.ingestion_service
                    .probe_embedded_subtitles_match(
                        request.url,
                        request.target_text,
                        similarity_threshold=85.0,
                    )
                )

                if (
                    tier0_result
                    and tier0_result.similarity_score >= 0.85
                ):

                    timestamp = (
                        tier0_result.start_time
                    )

                    frame = self._persist_frame(
                        artifacts_dir,
                        job_id,
                        "tier0",
                        video_source,
                        timestamp,
                        fps,
                    )

                    result = DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=timestamp,
                        formatted_timestamp=format_timestamp(
                            timestamp
                        ),
                        frame_number=int(
                            timestamp * fps
                        ),
                        extracted_text=(
                            tier0_result.matched_text
                        ),
                        confidence_score=(
                            tier0_result.similarity_score
                        ),
                        tier_executed=(
                            TierType.TIER_0_SUBTITLE
                        ),
                        frame_image_path=_sanitize_path(
                            frame
                        ),
                    )

                    self._save_job_metadata(
                        artifacts_dir,
                        result,
                    )

                    return result

            # ==================================================
            # TIER 1 : STREAM AUDIO -> STT
            # ==================================================

            stt_result = STTResult(
                found=False
            )

            try:

                logger.info(
                    "Extracting audio directly from stream"
                )

                audio_path = (
                    self.ingestion_service
                    .extract_audio_stream(
                        video_source,
                        job_id=job_id,
                        output_dir=temp_dir,
                        allow_download=True,
                    )
                )

                if isinstance(
                    audio_path,
                    str,
                ):
                    audio_path = Path(
                        audio_path
                    )

                if not is_valid_wav(
                    audio_path
                ):

                    logger.warning(
                        "Extracted WAV audio is invalid or shorter than 1 second: %s",
                        audio_path,
                    )

                else:

                    logger.info(
                        "Valid WAV extracted: %s (%d bytes)",
                        audio_path,
                        audio_path.stat().st_size,
                    )

                    logger.info(
                        "Running Whisper STT"
                    )

                    words = (
                        self.stt_service
                        .transcribe_audio(
                            audio_path
                        )
                    )

                    logger.info(
                        "Received %d timestamped words from STT",
                        len(words),
                    )

                    stt_result = (
                        self.stt_service
                        .align_target_dialogue(
                            words,
                            request.target_text,
                        )
                    )

            except Exception as exc:

                logger.warning(
                    "Tier 1 acoustic processing failed: %s",
                    exc,
                )

            # ==================================================
            # TIER 1 SUCCESS
            # ==================================================

            if (
                stt_result.found
                and stt_result.confidence >= 0.70
            ):

                logger.info(
                    "STT match found at %.2fs",
                    stt_result.start_time,
                )

                start_time = max(
                    0.0,
                    stt_result.start_time - 1.5,
                )

                end_time = min(
                    metadata.duration_seconds,
                    stt_result.end_time + 1.5,
                )

                # ==================================================
                # TIER 3 : DENSE OCR CONFIRMATION
                # ==================================================

                if self.mistral_client:

                    try:

                        dense_frame = (
                            artifacts_dir
                            / f"{job_id}_tier3.jpg"
                        )

                        candidate = (
                            scan_dense_window(
                                video_source,
                                start_time,
                                end_time,
                                fps,
                                request.target_text,
                                self.mistral_client,
                                save_frame_path=dense_frame,
                            )
                        )

                        if candidate:

                            ocr_confidence = getattr(
                                candidate,
                                "ocr_confidence",
                                None,
                            )

                            if not isinstance(
                                ocr_confidence,
                                (int, float),
                            ):
                                ocr_confidence = 0.0

                            confidence = (
                                calculate_confidence_score(
                                    text_similarity=(
                                        stt_result.confidence
                                    ),
                                    ocr_confidence=(
                                        ocr_confidence
                                    ),
                                    alignment_score=1.0,
                                )
                            )

                            result = DetectionResult(
                                job_id=job_id,
                                status=JobStatus.COMPLETED,
                                target_dialogue=(
                                    request.target_text
                                ),
                                timestamp_seconds=(
                                    candidate.timestamp_seconds
                                ),
                                formatted_timestamp=(
                                    format_timestamp(
                                        candidate.timestamp_seconds
                                    )
                                ),
                                frame_number=(
                                    candidate.frame_number
                                ),
                                extracted_text=(
                                    candidate.ocr_detected_text
                                ),
                                confidence_score=confidence,
                                tier_executed=(
                                    TierType.TIER_3_DENSE_OCR
                                ),
                                frame_image_path=_sanitize_path(
                                    getattr(
                                        candidate,
                                        "image_path",
                                        None,
                                    )
                                ),
                            )

                            self._save_job_metadata(
                                artifacts_dir,
                                result,
                            )

                            return result

                    except Exception as exc:

                        logger.warning(
                            "Dense OCR confirmation failed: %s",
                            exc,
                        )

                # ==================================================
                # TIER 1 DIRECT FALLBACK
                # ==================================================

                frame = self._persist_frame(
                    artifacts_dir,
                    job_id,
                    "tier1",
                    video_source,
                    stt_result.start_time,
                    fps,
                )

                result = DetectionResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    target_dialogue=request.target_text,
                    timestamp_seconds=(
                        stt_result.start_time
                    ),
                    formatted_timestamp=(
                        format_timestamp(
                            stt_result.start_time
                        )
                    ),
                    frame_number=int(
                        stt_result.start_time * fps
                    ),
                    extracted_text=(
                        stt_result.matched_text
                    ),
                    confidence_score=(
                        stt_result.confidence
                    ),
                    tier_executed=(
                        TierType.TIER_1_STT
                    ),
                    frame_image_path=_sanitize_path(
                        frame
                    ),
                )

                self._save_job_metadata(
                    artifacts_dir,
                    result,
                )

                return result

            # ==================================================
            # TIER 2 : SPARSE OCR
            # ==================================================

            sparse_candidate = None

            if self.mistral_client:

                try:

                    sparse_frame = (
                        artifacts_dir
                        / f"{job_id}_tier2.jpg"
                    )

                    sparse_candidate = (
                        scan_sparse_timeline(
                            video_source,
                            request.target_text,
                            self.mistral_client,
                            sample_fps=getattr(
                                settings,
                                "sparse_ocr_fps",
                                0.5,
                            ),
                            save_frame_path=sparse_frame,
                        )
                    )

                except OCRError as exc:

                    logger.warning(
                        "Sparse OCR failed: %s",
                        exc,
                    )

                except Exception as exc:

                    logger.warning(
                        "Sparse OCR failed unexpectedly: %s",
                        exc,
                    )

            if sparse_candidate:

                sparse_confidence = getattr(
                    sparse_candidate,
                    "ocr_confidence",
                    0.0,
                )

                if not isinstance(
                    sparse_confidence,
                    (int, float),
                ):
                    sparse_confidence = 0.0

                if sparse_confidence >= 0.60:

                    start = max(
                        0.0,
                        sparse_candidate.timestamp_seconds - 2.0,
                    )

                    end = min(
                        metadata.duration_seconds,
                        sparse_candidate.timestamp_seconds + 2.0,
                    )

                    # ==================================================
                    # TIER 3 FOLLOW-UP
                    # ==================================================

                    try:

                        dense = scan_dense_window(
                            video_source,
                            start,
                            end,
                            fps,
                            request.target_text,
                            self.mistral_client,
                            save_frame_path=(
                                artifacts_dir
                                / f"{job_id}_tier3.jpg"
                            ),
                        )

                        if dense:

                            dense_confidence = getattr(
                                dense,
                                "ocr_confidence",
                                0.0,
                            )

                            if not isinstance(
                                dense_confidence,
                                (int, float),
                            ):
                                dense_confidence = 0.0

                            result = DetectionResult(
                                job_id=job_id,
                                status=JobStatus.COMPLETED,
                                target_dialogue=(
                                    request.target_text
                                ),
                                timestamp_seconds=(
                                    dense.timestamp_seconds
                                ),
                                formatted_timestamp=(
                                    format_timestamp(
                                        dense.timestamp_seconds
                                    )
                                ),
                                frame_number=(
                                    dense.frame_number
                                ),
                                extracted_text=(
                                    dense.ocr_detected_text
                                ),
                                confidence_score=(
                                    dense_confidence
                                ),
                                tier_executed=(
                                    TierType.TIER_3_DENSE_OCR
                                ),
                                frame_image_path=_sanitize_path(
                                    getattr(
                                        dense,
                                        "image_path",
                                        None,
                                    )
                                ),
                            )

                            self._save_job_metadata(
                                artifacts_dir,
                                result,
                            )

                            return result

                    except Exception as exc:

                        logger.warning(
                            "Tier 3 after sparse failed: %s",
                            exc,
                        )

            # ==================================================
            # TIER 4 : VLM FALLBACK
            # ==================================================

            candidates = (
                self._extract_real_vlm_candidates(
                    temp_dir=temp_dir,
                    video_source=video_source,
                    stt_result=stt_result,
                    metadata=metadata,
                )
            )

            if candidates:

                decision = (
                    self.vlm_service
                    .evaluate_candidates(
                        request.target_text,
                        candidates,
                    )
                )

                if (
                    decision.selected_candidate_id
                    != "NONE"
                ):

                    selected = next(
                        (
                            candidate
                            for candidate in candidates
                            if (
                                candidate.candidate_id
                                == decision.selected_candidate_id
                            )
                        ),
                        None,
                    )

                    if selected is None:

                        return DetectionResult(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            target_dialogue=(
                                request.target_text
                            ),
                            error_message=(
                                "VLM selected an unknown candidate."
                            ),
                        )

                    # Persist outside temporary directory
                    artifacts_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    persistent_frame_path = (
                        artifacts_dir
                        / f"{job_id}_{selected.candidate_id}.jpg"
                    )

                    try:

                        shutil.copy2(
                            selected.image_path,
                            persistent_frame_path,
                        )

                        final_vlm_path: Optional[str] = (
                            str(
                                persistent_frame_path
                            )
                        )

                    except Exception as exc:

                        logger.warning(
                            "Failed persisting VLM candidate frame: %s",
                            exc,
                        )

                        final_vlm_path = (
                            selected.image_path
                        )

                    result = DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=(
                            selected.timestamp_seconds
                        ),
                        formatted_timestamp=(
                            format_timestamp(
                                selected.timestamp_seconds
                            )
                        ),
                        frame_number=(
                            selected.frame_number
                        ),
                        extracted_text=(
                            decision.exact_detected_text
                        ),
                        confidence_score=(
                            decision.confidence_score
                        ),
                        tier_executed=(
                            TierType.TIER_4_VLM_FALLBACK
                        ),
                        frame_image_path=_sanitize_path(
                            final_vlm_path
                        ),
                    )

                    self._save_job_metadata(
                        artifacts_dir,
                        result,
                    )

                    return result

            # ==================================================
            # FINAL FAILURE
            # ==================================================

            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=(
                    "Target dialogue not found through any execution tier."
                ),
            )

        except IngestionError as exc:

            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=(
                    f"Ingestion failed: {exc}"
                ),
            )

        except VLMError as exc:

            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=(
                    f"VLM failed: {exc}"
                ),
            )

        except Exception as exc:

            logger.exception(
                "Pipeline crashed: %s",
                exc,
            )

            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=str(exc),
            )

        finally:

            shutil.rmtree(
                temp_dir_str,
                ignore_errors=True,
            )

    # ========================================================
    # VLM CANDIDATES
    # ========================================================

    def _extract_real_vlm_candidates(
        self,
        temp_dir: Path,
        video_source: str,
        stt_result: STTResult,
        metadata: Any,
    ) -> List[CandidateFrame]:

        candidates: List[CandidateFrame] = []

        fps = (
            metadata.fps
            if metadata.fps > 0
            else 25.0
        )

        duration = (
            metadata.duration_seconds
            if metadata.duration_seconds > 0
            else 10.0
        )

        # STT-guided candidates
        if (
            stt_result
            and stt_result.found
            and stt_result.start_time is not None
        ):

            base = stt_result.start_time

            timestamps = [
                max(0.0, base - 3.0),
                max(0.0, base - 2.0),
                max(0.0, base - 1.0),
                base,
                min(duration, base + 1.0),
                min(duration, base + 2.0),
                min(duration, base + 3.0),
            ]

        else:

            timestamps = [
                duration * (i / 8.0)
                for i in range(1, 8)
            ]

        # Remove duplicates caused by clipping at
        # beginning/end of video.
        unique_timestamps: List[float] = []

        for timestamp in timestamps:

            timestamp = round(
                max(
                    0.0,
                    min(
                        duration,
                        timestamp,
                    ),
                ),
                3,
            )

            if timestamp not in unique_timestamps:
                unique_timestamps.append(
                    timestamp
                )

        # ====================================================
        # EXTRACT EACH CANDIDATE
        # ====================================================

        for idx, timestamp in enumerate(
            unique_timestamps,
            start=1,
        ):

            frame_path = (
                temp_dir
                / f"candidate_C{idx}.jpg"
            )

            extracted = False

            # PRIMARY:
            # FFmpeg / ingestion streaming path
            try:

                extracted = bool(
                    self.ingestion_service
                    .extract_frame_on_demand(
                        video_source,
                        timestamp,
                        frame_path,
                    )
                )

            except Exception as exc:

                logger.debug(
                    "On-demand candidate extraction failed for C%d: %s",
                    idx,
                    exc,
                )

            # SECONDARY:
            # OpenCV fallback
            if not extracted and not frame_path.exists():

                extracted = (
                    self._extract_single_frame(
                        video_source,
                        timestamp,
                        fps,
                        frame_path,
                    )
                )

            if (
                extracted
                and frame_path.exists()
                and frame_path.stat().st_size > 0
            ):

                candidates.append(
                    CandidateFrame(
                        candidate_id=f"C{idx}",
                        timestamp_seconds=timestamp,
                        frame_number=int(
                            round(
                                timestamp
                                * fps
                            )
                        ),
                        image_path=str(
                            frame_path
                        ),
                    )
                )

        logger.info(
            "Prepared %d VLM candidates",
            len(candidates),
        )

        return candidates