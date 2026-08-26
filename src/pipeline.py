import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import cv2

from src.fallback_vlm import VLMArbiterService, VLMError
from src.ingestion import IngestionError, StreamIngestionService
from src.models.schemas import (
    CandidateFrame,
    DetectionResult,
    JobRequest,
    JobStatus,
    STTResult,
    TierType,
    VLMDecision,
)
from src.primary_ocr import (
    OCRError,
    scan_dense_window,
    scan_sparse_timeline,
)
from src.primary_stt import SpeechToTextService

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format decimal seconds into HH:MM:SS.sss representation strictly handling non-finite inputs."""
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
    """
    Confidence Fusion Engine:
    Blends text similarity, OCR confidence, and temporal/STT alignment into a normalized score.
    Formula: C_final = w_sim * S_text + w_ocr * C_ocr + w_align * A_temporal
    """
    raw_score = (w_sim * text_similarity) + (w_ocr * ocr_confidence) + (w_align * alignment_score)
    return round(max(0.0, min(1.0, raw_score)), 4)


class PipelineOrchestrator:
    """
    Hybrid Multi-Tier Pipeline Orchestrator.
    Hierarchical execution flow:
      - Tier 0: Soft Subtitle Cue Matching
      - Tier 1: Acoustic STT Probing
      - Tier 3: Dense Visual Onset Scan (triggered by Tier 1 candidate window)
      - Tier 2: Sparse Timeline OCR Scan (0.5 FPS) -> Tier 3 Dense Scan
      - Tier 4: VLM Bounded Candidate Arbitration
      - Fallback: Tier 1 Acoustic-Only Match (when visual localization cannot be performed)
    """

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

    def run(self, job_id: str, request: JobRequest) -> DetectionResult:
        """Execute end-to-end multi-tier dialogue detection for a job request."""
        temp_dir_str = tempfile.mkdtemp(prefix=f"quest1_{job_id}_")
        temp_dir = Path(temp_dir_str)
        logger.info("Executing job %s for URL: %s", job_id, request.url)

        try:
            metadata = self.ingestion_service.probe_metadata(request.url)
            fps = metadata.fps if metadata.fps > 0 else 25.0
            video_source: Optional[str] = metadata.stream_path if metadata.stream_path else None

            # --- TIER 0: Embedded Subtitle Probing ---
            if metadata.has_subtitles:
                tier0_result = self.ingestion_service.probe_embedded_subtitles_match(
                    request.url, request.target_text, similarity_threshold=85.0
                )
                if tier0_result and tier0_result.similarity_score >= 0.85:
                    target_time = tier0_result.start_time
                    frame_num = int(round(target_time * fps))
                    return DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=target_time,
                        formatted_timestamp=format_timestamp(target_time),
                        frame_number=frame_num,
                        extracted_text=tier0_result.matched_text,
                        confidence_score=tier0_result.similarity_score,
                        tier_executed=TierType.TIER_0_SUBTITLE,
                    )

            # --- TIER 1: Acoustic Path Probing (Whisper STT) ---
            stt_result = STTResult(found=False)
            try:
                audio_path = self.ingestion_service.extract_audio_stream(
                    request.url, job_id=job_id, output_dir=temp_dir
                )
                if isinstance(audio_path, str):
                    audio_path = Path(audio_path)

                words = self.stt_service.transcribe_audio(audio_path)
                stt_result = self.stt_service.align_target_dialogue(words, request.target_text)
            except Exception as stt_err:
                logger.warning("Tier 1 Acoustic processing failed or found no audio: %s", stt_err)

            # --- TIER 3: Attempt Dense OCR Onset Scan via STT Candidate Window ---
            if stt_result.found and stt_result.confidence >= 0.70:
                t_start = max(0.0, stt_result.start_time - 1.5)
                t_end = min(metadata.duration_seconds, stt_result.end_time + 1.5)

                if video_source and self.mistral_client:
                    try:
                        dense_cand = scan_dense_window(
                            video_source, t_start, t_end, fps, request.target_text, self.mistral_client
                        )
                        if dense_cand:
                            ocr_conf = (
                                dense_cand.ocr_confidence
                                if dense_cand.ocr_confidence is not None
                                else 0.0
                            )
                            fused_conf = calculate_confidence_score(
                                text_similarity=stt_result.confidence,
                                ocr_confidence=ocr_conf,
                                alignment_score=1.0,
                            )
                            return DetectionResult(
                                job_id=job_id,
                                status=JobStatus.COMPLETED,
                                target_dialogue=request.target_text,
                                timestamp_seconds=dense_cand.timestamp_seconds,
                                formatted_timestamp=format_timestamp(dense_cand.timestamp_seconds),
                                frame_number=dense_cand.frame_number,
                                extracted_text=dense_cand.ocr_detected_text,
                                confidence_score=fused_conf,
                                tier_executed=TierType.TIER_3_DENSE_OCR,
                            )
                    except OCRError as ocr_err:
                        logger.warning("Tier 3 Dense OCR scan failed following STT: %s", ocr_err)

            # --- TIER 2: Attempt Sparse Timeline OCR ---
            sparse_cand = None
            if video_source and self.mistral_client:
                try:
                    sparse_cand = scan_sparse_timeline(
                        video_source, request.target_text, self.mistral_client, sample_fps=0.5
                    )
                except OCRError as ocr_err:
                    logger.warning("Tier 2 Sparse OCR scan failed: %s", ocr_err)

            if sparse_cand and (sparse_cand.ocr_confidence or 0.0) >= 0.60:
                t_start = max(0.0, sparse_cand.timestamp_seconds - 2.0)
                t_end = min(metadata.duration_seconds, sparse_cand.timestamp_seconds + 2.0)

                try:
                    dense_cand = scan_dense_window(
                        video_source, t_start, t_end, fps, request.target_text, self.mistral_client
                    )
                    if dense_cand:
                        return DetectionResult(
                            job_id=job_id,
                            status=JobStatus.COMPLETED,
                            target_dialogue=request.target_text,
                            timestamp_seconds=dense_cand.timestamp_seconds,
                            formatted_timestamp=format_timestamp(dense_cand.timestamp_seconds),
                            frame_number=dense_cand.frame_number,
                            extracted_text=dense_cand.ocr_detected_text,
                            confidence_score=dense_cand.ocr_confidence,
                            tier_executed=TierType.TIER_3_DENSE_OCR,
                        )
                except OCRError as ocr_err:
                    logger.warning("Tier 3 Dense OCR scan failed following Sparse OCR: %s", ocr_err)

                if (sparse_cand.ocr_confidence or 0.0) >= 0.80:
                    return DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=sparse_cand.timestamp_seconds,
                        formatted_timestamp=format_timestamp(sparse_cand.timestamp_seconds),
                        frame_number=sparse_cand.frame_number,
                        extracted_text=sparse_cand.ocr_detected_text,
                        confidence_score=sparse_cand.ocr_confidence,
                        tier_executed=TierType.TIER_2_SPARSE_OCR,
                    )

            # --- TIER 4: Attempt VLM Arbiter Fallback ---
            candidates = self._extract_real_vlm_candidates(temp_dir, video_source, stt_result, metadata)
            if candidates:
                vlm_decision = self.vlm_service.evaluate_candidates(request.target_text, candidates)
                if vlm_decision.selected_candidate_id != "NONE":
                    selected_cand = next(
                        c for c in candidates if c.candidate_id == vlm_decision.selected_candidate_id
                    )
                    artifacts_dir = Path("artifacts") / job_id
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    persistent_frame_path = artifacts_dir / f"{job_id}_{selected_cand.candidate_id}.jpg"
                    shutil.copy2(selected_cand.image_path, persistent_frame_path)

                    return DetectionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        target_dialogue=request.target_text,
                        timestamp_seconds=selected_cand.timestamp_seconds,
                        formatted_timestamp=format_timestamp(selected_cand.timestamp_seconds),
                        frame_number=selected_cand.frame_number,
                        extracted_text=vlm_decision.exact_detected_text,
                        confidence_score=vlm_decision.confidence_score,
                        tier_executed=TierType.TIER_4_VLM_FALLBACK,
                        frame_image_path=str(persistent_frame_path),
                    )

            # --- TIER 1 ACOUSTIC FALLBACK ---
            if stt_result.found and stt_result.start_time is not None:
                frame_num = int(round(stt_result.start_time * fps))
                return DetectionResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    target_dialogue=request.target_text,
                    timestamp_seconds=stt_result.start_time,
                    formatted_timestamp=format_timestamp(stt_result.start_time),
                    frame_number=frame_num,
                    extracted_text=stt_result.matched_text,
                    confidence_score=stt_result.confidence,
                    tier_executed=TierType.TIER_1_STT,
                )

            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message="Target dialogue could not be located across any execution tier.",
            )

        except IngestionError as ing_err:
            logger.error("Ingestion error in job %s: %s", job_id, ing_err)
            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=f"Ingestion failed: {ing_err}",
            )
        except VLMError as vlm_err:
            logger.error("VLM error in job %s: %s", job_id, vlm_err)
            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=f"VLM arbitration failed: {vlm_err}",
            )
        except Exception as exc:
            logger.error("Pipeline execution failed for job %s: %s", job_id, exc, exc_info=True)
            return DetectionResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                target_dialogue=request.target_text,
                error_message=str(exc),
            )
        finally:
            shutil.rmtree(temp_dir_str, ignore_errors=True)

    def _extract_real_vlm_candidates(
        self, temp_dir: Path, video_source: Optional[str], stt_result: STTResult, metadata: Any
    ) -> List[CandidateFrame]:
        """Extract REAL video frame images using OpenCV at candidate timestamps onto disk for Tier 4."""
        if not video_source:
            logger.warning("No valid direct media stream or local video file available for candidate extraction.")
            return []

        candidates: List[CandidateFrame] = []
        fps = metadata.fps if metadata.fps > 0 else 25.0
        duration = metadata.duration_seconds if metadata.duration_seconds > 0 else 10.0

        if stt_result and stt_result.found and stt_result.start_time is not None:
            t0 = stt_result.start_time
            timestamps = [max(0.0, t0 - 1.0), t0, min(duration, t0 + 1.0)]
        else:
            timestamps = [duration * 0.25, duration * 0.50, duration * 0.75]

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.warning("VideoCapture could not open media source: %s", video_source)
            return []

        try:
            for idx, ts in enumerate(timestamps, start=1):
                frame_num = int(round(ts * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_path = temp_dir / f"candidate_C{idx}.jpg"
                success = cv2.imwrite(str(frame_path), frame)
                if not success or not frame_path.exists():
                    continue

                candidates.append(
                    CandidateFrame(
                        candidate_id=f"C{idx}",
                        timestamp_seconds=round(ts, 3),
                        frame_number=frame_num,
                        image_path=str(frame_path),
                    )
                )
        finally:
            cap.release()

        return candidates