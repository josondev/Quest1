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
    return round(max(0.0, min(1.0, score)), 4)


def _sanitize_path(val: Any) -> Optional[str]:
    if isinstance(val, (str, Path)) and not hasattr(val, "_mock_return_value"):
        value = str(val).strip()
        return value if value else None
    return None


def is_valid_wav(path: Union[str, Path]) -> bool:
    """Validates that extracted audio file exists, is non-empty, and has a valid WAV header."""
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 44:
            return False
        with wave.open(str(p), "rb") as wf:
            return wf.getnframes() > 0 or p.stat().st_size > 44
    except Exception as exc:
        logger.debug("WAV validation check failed for %s: %s", path, exc)
        return False


class PipelineOrchestrator:
    def __init__(
        self,
        ingestion_service: Optional[StreamIngestionService] = None,
        stt_service: Optional[SpeechToTextService] = None,
        vlm_service: Optional[VLMArbiterService] = None,
        mistral_client: Optional[Any] = None,
    ):
        self.ingestion_service = ingestion_service or StreamIngestionService()
        self.stt_service = stt_service or SpeechToTextService()
        self.vlm_service = vlm_service or VLMArbiterService()
        self.mistral_client = mistral_client

    def _extract_single_frame(
        self,
        video_source: str,
        timestamp_seconds: float,
        fps: float,
        dest_path: Path,
    ) -> bool:
        """Extract frame via OpenCV. Strictly restricted to local video files to guard Windows crashes."""
        if video_source.startswith("http"):
            logger.debug("Skipping OpenCV frame extraction for remote URL: %s", video_source[:80])
            return False

        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                return False

            frame_number = int(round(timestamp_seconds * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret or frame is None:
                return False

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            return bool(cv2.imwrite(str(dest_path), frame))
        except Exception as exc:
            logger.debug("OpenCV frame extraction failed for %s: %s", video_source, exc)
            return False
        finally:
            try:
                if "cap" in locals() and cap is not None:
                    cap.release()
            except Exception:
                pass

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

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        frame_path = artifacts_dir / f"{job_id}_{tier_label}.jpg"

        if self.ingestion_service.extract_frame_on_demand(
            video_source, timestamp_seconds, frame_path
        ) and frame_path.exists():
            logger.info("Frame extracted from stream at %.2fs", timestamp_seconds)
            return str(frame_path)

        if self._extract_single_frame(
            video_source, timestamp_seconds, fps, frame_path
        ):
            return str(frame_path)

        logger.warning("Unable to extract frame at %.2fs", timestamp_seconds)
        return None

    def _save_job_metadata(
        self, artifacts_dir: Path, result: DetectionResult
    ) -> None:
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = artifacts_dir / "metadata.json"
            metadata_file.write_text(
                json.dumps(result.model_dump(), indent=2)
            )
        except Exception as e:
            logger.warning("Metadata save failed: %s", e)

    def run(self, job_id: str, request: JobRequest) -> DetectionResult:
        temp_dir_str = tempfile.mkdtemp(prefix=f"quest1_{job_id}_")
        temp_dir = Path(temp_dir_str)

        artifacts_base = getattr(
            settings,
            "artifacts_dir",
            getattr(settings, "artifact_storage_dir", Path("artifacts")),
        )
        artifacts_dir = Path(artifacts_base) / job_id

        logger.info("Executing job %s", job_id)

        try:
            metadata = self.ingestion_service.probe_metadata(request.url)
            fps = metadata.fps if metadata.fps > 0 else 25.0
            video_source = metadata.stream_path or request.url

            if metadata.is_local:
                logger.info("Processing local video: %s", video_source)
            else:
                logger.info(
                    "Processing remote stream: %s", str(video_source or "")[:120]
                )

            # --- TIER 0 : SUBTITLE MATCH ---
            if metadata.has_subtitles:
                tier0_result = (
                    self.ingestion_service.probe_embedded_subtitles_match(
                        request.url,
                        request.target_text,
                        similarity_threshold=85.0,
                    )
                )

                if (
                    tier0_result
                    and tier0_result.similarity_score >= 0.85
                ):
                    timestamp = tier0_result.start_time
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
                        formatted_timestamp=format_timestamp(timestamp),
                        frame_number=int(round(timestamp * fps)),
                        extracted_text=tier0_result.matched_text,
                        confidence_score=tier0_result.similarity_score,
                        tier_executed=TierType.TIER_0_SUBTITLE,
                        frame_image_path=_sanitize_path(frame),
                    )

                    self._save_job_metadata(artifacts_dir, result)
                    return result

            # --- TIER 1 : STREAM AUDIO -> WHISPER STT ---
            stt_result = STTResult(found=False)

            try:
                logger.info("Extracting audio directly from stream")
                audio_path = self.ingestion_service.extract_audio_stream(
                    video_source,
                    job_id=job_id,
                    output_dir=temp_dir,
                    allow_download=True,
                )

                # Re-route video_source to downloaded local file cache if present
                local_mp4 = temp_dir / f"{job_id}_video.mp4"
                if local_mp4.exists() and local_mp4.stat().st_size > 0:
                    video_source = str(local_mp4)
                    logger.info("Updated video_source to local file cache: %s", video_source)

                if isinstance(audio_path, str):
                    audio_path = Path(audio_path)

                if is_valid_wav(audio_path):
                    logger.info("Running Whisper STT on valid audio")
                    words = self.stt_service.transcribe_audio(audio_path)
                    stt_result = self.stt_service.align_target_dialogue(
                        words, request.target_text
                    )
                else:
                    logger.warning("Extracted WAV audio is invalid or silent. Skipping STT alignment.")
            except Exception as e:
                logger.warning("Tier 1 failed: %s", e)

            if stt_result.found and stt_result.confidence >= 0.70:
                logger.info("STT match found at %.2fs", stt_result.start_time)

                # --- TIER 3 : Dense OCR confirmation ---
                start_time = max(0.0, stt_result.start_time - 1.5)
                end_time = min(
                    metadata.duration_seconds, stt_result.end_time + 1.5
                )

                if self.mistral_client:
                    try:
                        dense_frame = artifacts_dir / f"{job_id}_tier3.jpg"
                        candidate = scan_dense_window(
                            video_source,
                            start_time,
                            end_time,
                            fps,
                            request.target_text,
                            self.mistral_client,
                            save_frame_path=dense_frame,
                        )

                        if candidate:
                            confidence = calculate_confidence_score(
                                stt_result.confidence,
                                getattr(candidate, "ocr_confidence", 0.8),
                                1.0,
                            )

                            result = DetectionResult(
                                job_id=job_id,
                                status=JobStatus.COMPLETED,
                                target_dialogue=request.target_text,
                                timestamp_seconds=candidate.timestamp_seconds,
                                formatted_timestamp=format_timestamp(
                                    candidate.timestamp_seconds
                                ),
                                frame_number=candidate.frame_number,
                                extracted_text=candidate.ocr_detected_text,
                                confidence_score=confidence,
                                tier_executed=TierType.TIER_3_DENSE_OCR,
                                frame_image_path=_sanitize_path(
                                    candidate.image_path
                                ),
                            )

                            self._save_job_metadata(artifacts_dir, result)
                            return result
                    except Exception as e:
                        logger.warning("Dense OCR confirmation failed: %s", e)

                # STT ONLY fallback
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
                    timestamp_seconds=stt_result.start_time,
                    formatted_timestamp=format_timestamp(
                        stt_result.start_time
                    ),
                    frame_number=int(round(stt_result.start_time * fps)),
                    extracted_text=stt_result.matched_text,
                    confidence_score=stt_result.confidence,
                    tier_executed=TierType.TIER_1_STT,
                    frame_image_path=_sanitize_path(frame),
                )

                self._save_job_metadata(artifacts_dir, result)
                return result

            # --- TIER 2 : SPARSE OCR STREAM SCAN ---
            sparse_candidate = None

            if self.mistral_client:
                try:
                    sparse_frame = artifacts_dir / f"{job_id}_tier2.jpg"
                    sparse_candidate = scan_sparse_timeline(
                        video_source,
                        request.target_text,
                        self.mistral_client,
                        sample_fps=getattr(settings, "sparse_ocr_fps", 0.5),
                        save_frame_path=sparse_frame,
                    )
                except Exception as e:
                    logger.warning("Sparse OCR failed: %s", e)

            if sparse_candidate:
                confidence = getattr(sparse_candidate, "ocr_confidence", 0.0)

                if confidence >= 0.60:
                    start = max(0.0, sparse_candidate.timestamp_seconds - 2.0)
                    end = min(
                        metadata.duration_seconds,
                        sparse_candidate.timestamp_seconds + 2.0,
                    )

                    # --- TIER 3 FOLLOW-UP ---
                    try:
                        dense = scan_dense_window(
                            video_source,
                            start,
                            end,
                            fps,
                            request.target_text,
                            self.mistral_client,
                            save_frame_path=(
                                artifacts_dir / f"{job_id}_tier3.jpg"
                            ),
                        )

                        if dense:
                            result = DetectionResult(
                                job_id=job_id,
                                status=JobStatus.COMPLETED,
                                target_dialogue=request.target_text,
                                timestamp_seconds=dense.timestamp_seconds,
                                formatted_timestamp=format_timestamp(
                                    dense.timestamp_seconds
                                ),
                                frame_number=dense.frame_number,
                                extracted_text=dense.ocr_detected_text,
                                confidence_score=dense.ocr_confidence,
                                tier_executed=TierType.TIER_3_DENSE_OCR,
                                frame_image_path=_sanitize_path(
                                    dense.image_path
                                ),
                            )

                            self._save_job_metadata(artifacts_dir, result)
                            return result
                    except Exception as e:
                        logger.warning("Tier3 after sparse failed: %s", e)

                    # Tier 2 standalone success path
                    frame = self._persist_frame(
                        artifacts_dir,
                        job_id,
                        "tier2",
                        video_source,
                        sparse_candidate.timestamp_seconds,
                        fps,
                    )
                    result = DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=sparse_candidate.timestamp_seconds,
                        formatted_timestamp=format_timestamp(
                            sparse_candidate.timestamp_seconds
                        ),
                        frame_number=int(round(
                            sparse_candidate.timestamp_seconds * fps
                        )),
                        extracted_text=getattr(
                            sparse_candidate, "ocr_detected_text", ""
                        ),
                        confidence_score=confidence,
                        tier_executed=TierType.TIER_2_SPARSE_OCR,
                        frame_image_path=_sanitize_path(frame),
                    )
                    self._save_job_metadata(artifacts_dir, result)
                    return result

            # --- TIER 4 : VLM FALLBACK ---
            candidates = self._extract_real_vlm_candidates(
                temp_dir,
                video_source,
                stt_result,
                metadata,
            )

            if candidates:
                decision = self.vlm_service.evaluate_candidates(
                    request.target_text, candidates
                )

                if decision.selected_candidate_id != "NONE":
                    selected = next(
                        c
                        for c in candidates
                        if c.candidate_id == decision.selected_candidate_id
                    )

                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    persistent_frame_path = (
                        artifacts_dir / f"{job_id}_{selected.candidate_id}.jpg"
                    )
                    try:
                        shutil.copy2(selected.image_path, persistent_frame_path)
                        final_vlm_path: Optional[str] = str(persistent_frame_path)
                    except Exception as copy_err:
                        logger.warning(
                            "Failed persisting VLM candidate frame artifact: %s", copy_err
                        )
                        final_vlm_path = selected.image_path

                    result = DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=selected.timestamp_seconds,
                        formatted_timestamp=format_timestamp(
                            selected.timestamp_seconds
                        ),
                        frame_number=selected.frame_number,
                        extracted_text=decision.exact_detected_text,
                        confidence_score=decision.confidence_score,
                        tier_executed=TierType.TIER_4_VLM_FALLBACK,
                        frame_image_path=_sanitize_path(final_vlm_path),
                    )

                    self._save_job_metadata(artifacts_dir, result)
                    return result

            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=(
                    "Target dialogue not found through any execution tier."
                ),
            )

        except IngestionError as e:
            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=f"Ingestion failed: {e}",
            )
        except VLMError as e:
            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=f"VLM failed: {e}",
            )
        except Exception as e:
            logger.exception("Pipeline crashed: %s", e)
            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=str(e),
            )
        finally:
            shutil.rmtree(temp_dir_str, ignore_errors=True)

    def _extract_real_vlm_candidates(
        self, temp_dir: Path, video_source: Optional[str], stt_result: STTResult, metadata: Any
    ) -> List[CandidateFrame]:
        candidates = []
        if not video_source:
            return candidates

        fps = metadata.fps if metadata.fps > 0 else 25.0
        duration = metadata.duration_seconds if metadata.duration_seconds > 0 else 10.0

        if stt_result and stt_result.found and stt_result.start_time is not None:
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
            timestamps = [duration * (i / 8.0) for i in range(1, 8)]

        for idx, ts in enumerate(timestamps, start=1):
            frame_path = temp_dir / f"candidate_C{idx}.jpg"

            self.ingestion_service.extract_frame_on_demand(
                video_source, ts, frame_path
            )

            if not frame_path.exists():
                self._extract_single_frame(video_source, ts, fps, frame_path)

            if frame_path.exists():
                candidates.append(
                    CandidateFrame(
                        candidate_id=f"C{idx}",
                        timestamp_seconds=round(ts, 3),
                        frame_number=int(round(ts * fps)),
                        image_path=str(frame_path),
                    )
                )

        return candidates