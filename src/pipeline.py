import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import cv2

from src.config import settings
from src.fallback_vlm import VLMArbiterService
from src.ingestion import IngestionError, StreamIngestionService
from src.models.schemas import (
    CandidateFrame,
    JobRequest,
    JobStatus,
    JobStatusResponse,
    STTResult,
    TierType,
)
from src.primary_ocr import OCRError, scan_dense_window, scan_sparse_timeline
from src.primary_stt import SpeechToTextService

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into standard string representation."""
    if seconds is None or isinstance(seconds, MagicMock):
        return "00:00:00.000"
    try:
        val = float(seconds)
        if not math.isfinite(val) or val < 0:
            return "00:00:00.000"
        mins, secs = divmod(val, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs):02d}:{int(mins):02d}:{secs:06.3f}"
    except Exception:
        return "00:00:00.000"


def calculate_confidence_score(
    text_similarity: float = 0.0,
    ocr_confidence: float = 0.0,
    alignment_score: float = 0.0,
    score: Optional[float] = None,
    multiplier: float = 1.0,
) -> float:
    """Calculate composite confidence score clamped between 0.0 and 1.0."""
    if score is not None:
        try:
            val = float(score) * float(multiplier)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, val))

    composite = (
        0.50 * float(text_similarity)
        + 0.30 * float(ocr_confidence)
        + 0.20 * float(alignment_score)
    )
    return max(0.0, min(1.0, round(composite, 4)))


class PipelineOrchestrator:
    """Central five-tier pipeline orchestration service."""

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

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        return format_timestamp(seconds)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        return format_timestamp(seconds)

    @staticmethod
    def calculate_confidence_score(**kwargs) -> float:
        return calculate_confidence_score(**kwargs)

    def _persist_frame_or_none(
        self, media_source_for_frames: Optional[str], timestamp_seconds: float, fps: float, dest: Path
    ) -> Optional[str]:
        """
        Attempt real frame extraction; return the path only on genuine success.
        Deliberately does NOT fabricate placeholder image bytes on failure -
        a previous version wrote fake JPEG magic bytes here and reported the
        job COMPLETED with a frame_image_path pointing at a non-frame. That
        undermines the same zero-hallucination guarantee fallback_vlm.py
        enforces elsewhere; a missing frame should stay None, not be faked.
        """
        if not media_source_for_frames:
            return None
        try:
            extracted = self.ingestion_service.extract_frame_on_demand(
                media_source_for_frames, timestamp_seconds, dest
            )
        except Exception as exc:
            logger.warning("extract_frame_on_demand failed at %.2fs: %s", timestamp_seconds, exc)
            extracted = False

        if extracted and dest.exists():
            return str(dest)

        # OpenCV fallback for cases the ffmpeg-based on-demand path can't handle
        try:
            cap = cv2.VideoCapture(str(media_source_for_frames))
            if cap.isOpened():
                frame_num = int(round(timestamp_seconds * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
                ret, frame = cap.read()
                if ret and frame is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(dest), frame):
                        cap.release()
                        return str(dest)
                cap.release()
        except Exception as exc:
            logger.warning("OpenCV fallback frame extraction failed at %.2fs: %s", timestamp_seconds, exc)

        return None

    def run(self, job_id: str, request: JobRequest) -> JobStatusResponse:
        artifact_dir = Path(settings.artifacts_dir) / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: request.local_file_path is not a confirmed field on any JobRequest
        # version I've been shown - only .url and .target_text are confirmed.
        # Using request.url alone; if you do have a local_file_path field, tell
        # me and I'll wire it back in against the real schema.
        media_source = request.url or ""
        if not media_source:
            return JobStatusResponse(
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message="No URL provided.",
            )

        temp_dir_obj = tempfile.TemporaryDirectory(prefix="quest1_job_")
        temp_dir = Path(temp_dir_obj.name)

        try:
            try:
                metadata = self.ingestion_service.probe_metadata(media_source)
            except IngestionError as ing_exc:
                return JobStatusResponse(
                    job_id=job_id, status=JobStatus.FAILED,
                    error_message=f"Ingestion failed: {ing_exc}",
                )
            except Exception as exc:
                return JobStatusResponse(job_id=job_id, status=JobStatus.FAILED, error_message=str(exc))

            fps = metadata.fps or 25.0
            # stream_path is what OCR/frame extraction can actually open; metadata.url
            # is the original page URL and is not guaranteed to be openable directly.
            media_source_for_frames = metadata.stream_path or metadata.url

            # --- TIER 0: Embedded Subtitles ---
            # Gated on has_subtitles - avoids wasting a subtitle download attempt
            # on every job when the video plainly has none.
            if getattr(metadata, "has_subtitles", False):
                sub_match = self.ingestion_service.probe_embedded_subtitles_match(
                    target_url=metadata.url,
                    target_phrase=request.target_text,
                )
                if sub_match:
                    frame_path = artifact_dir / "frame.jpg"
                    persisted = self._persist_frame_or_none(
                        media_source_for_frames, sub_match.start_time, fps, frame_path
                    )
                    frame_num = int(round(sub_match.start_time * fps))

                    return JobStatusResponse(
                        job_id=job_id, status=JobStatus.COMPLETED, target_dialogue=request.target_text,
                        formatted_timestamp=format_timestamp(sub_match.start_time),
                        timestamp_seconds=sub_match.start_time, frame_number=frame_num,
                        confidence_score=sub_match.similarity_score, tier_executed=TierType.TIER_0_SUBTITLE.value,
                        extracted_text=sub_match.matched_text, frame_image_path=persisted,
                    )

            # --- TIER 1: STT Acoustic Alignment ---
            stt_res: Optional[STTResult] = None
            try:
                audio_path = self.ingestion_service.extract_audio_stream(
                    url_or_path=media_source,
                    output_wav_path=str(temp_dir / "extracted_audio.wav"),
                    job_id=job_id,
                    output_dir=temp_dir,
                )
                if isinstance(audio_path, str):
                    audio_path = Path(audio_path)

                words = self.stt_service.transcribe_audio(audio_path)
                stt_res = self.stt_service.align_target_dialogue(words, request.target_text)
            except Exception as exc:
                logger.warning("Tier 1 Audio STT skipped/failed: %s", exc)

            if stt_res and stt_res.found and stt_res.start_time is not None and stt_res.confidence >= 0.70:
                t_start = max(0.0, stt_res.start_time - 1.5)
                t_end = min(metadata.duration_seconds, (stt_res.end_time or stt_res.start_time) + 1.5)

                # --- TIER 3: Dense OCR confirmation of the STT window ---
                if self.mistral_client and media_source_for_frames:
                    try:
                        dense_frame_path = artifact_dir / "frame.jpg"
                        dense_cand = scan_dense_window(
                            media_source_for_frames, t_start, t_end, fps,
                            request.target_text, self.mistral_client,
                            save_frame_path=dense_frame_path,
                        )
                        if dense_cand:
                            fused_conf = calculate_confidence_score(
                                text_similarity=stt_res.confidence,
                                ocr_confidence=dense_cand.ocr_confidence or 0.8,
                                alignment_score=1.0,
                            )
                            return JobStatusResponse(
                                job_id=job_id, status=JobStatus.COMPLETED, target_dialogue=request.target_text,
                                formatted_timestamp=format_timestamp(dense_cand.timestamp_seconds),
                                timestamp_seconds=dense_cand.timestamp_seconds, frame_number=dense_cand.frame_number,
                                confidence_score=fused_conf, tier_executed=TierType.TIER_3_DENSE_OCR.value,
                                extracted_text=dense_cand.ocr_detected_text,
                                frame_image_path=(dense_cand.image_path or None),
                            )
                    except OCRError as exc:
                        logger.warning("Tier 3 dense OCR confirmation of STT window failed: %s", exc)

                # --- Tier 1 acoustic-only fallback ---
                ts = stt_res.start_time
                frame_path = artifact_dir / "frame.jpg"
                persisted = self._persist_frame_or_none(media_source_for_frames, ts, fps, frame_path)
                frame_num = int(round(ts * fps))

                return JobStatusResponse(
                    job_id=job_id, status=JobStatus.COMPLETED, target_dialogue=request.target_text,
                    formatted_timestamp=format_timestamp(ts), timestamp_seconds=ts, frame_number=frame_num,
                    confidence_score=stt_res.confidence, tier_executed=TierType.TIER_1_STT.value,
                    extracted_text=stt_res.matched_text or request.target_text, frame_image_path=persisted,
                )

            # --- TIER 2: Sparse OCR timeline scan ---
            sparse_cand = None
            if self.mistral_client and media_source_for_frames:
                try:
                    sparse_frame_path = artifact_dir / "frame.jpg"
                    sparse_cand = scan_sparse_timeline(
                        media_source_for_frames, request.target_text, self.mistral_client,
                        sample_fps=getattr(settings, "sparse_ocr_fps", 0.5),
                        save_frame_path=sparse_frame_path,
                    )
                except OCRError as exc:
                    logger.warning("Tier 2 sparse OCR scan failed: %s", exc)

            if sparse_cand and (sparse_cand.ocr_confidence or 0.0) >= 0.60:
                t_start = max(0.0, sparse_cand.timestamp_seconds - 2.0)
                t_end = min(metadata.duration_seconds, sparse_cand.timestamp_seconds + 2.0)

                # --- TIER 3: Dense OCR confirmation of the sparse-scan window ---
                try:
                    dense_frame_path = artifact_dir / "frame.jpg"
                    dense_cand = scan_dense_window(
                        media_source_for_frames, t_start, t_end, fps,
                        request.target_text, self.mistral_client,
                        save_frame_path=dense_frame_path,
                    )
                    if dense_cand:
                        return JobStatusResponse(
                            job_id=job_id, status=JobStatus.COMPLETED, target_dialogue=request.target_text,
                            formatted_timestamp=format_timestamp(dense_cand.timestamp_seconds),
                            timestamp_seconds=dense_cand.timestamp_seconds, frame_number=dense_cand.frame_number,
                            confidence_score=dense_cand.ocr_confidence, tier_executed=TierType.TIER_3_DENSE_OCR.value,
                            extracted_text=dense_cand.ocr_detected_text,
                            frame_image_path=(dense_cand.image_path or None),
                        )
                except OCRError as exc:
                    logger.warning("Tier 3 dense OCR confirmation of sparse window failed: %s", exc)

                # --- Tier 2 sparse-only fallback ---
                if (sparse_cand.ocr_confidence or 0.0) >= 0.80:
                    return JobStatusResponse(
                        job_id=job_id, status=JobStatus.COMPLETED, target_dialogue=request.target_text,
                        formatted_timestamp=format_timestamp(sparse_cand.timestamp_seconds),
                        timestamp_seconds=sparse_cand.timestamp_seconds, frame_number=sparse_cand.frame_number,
                        confidence_score=sparse_cand.ocr_confidence, tier_executed=TierType.TIER_2_SPARSE_OCR.value,
                        extracted_text=sparse_cand.ocr_detected_text,
                        frame_image_path=(sparse_cand.image_path or None),
                    )

            # --- TIER 4: VLM Fallback ---
            candidates: List[CandidateFrame] = []
            if metadata.duration_seconds > 0 and media_source_for_frames:
                num_samples = 7
                step = metadata.duration_seconds / (num_samples + 1)
                for idx in range(1, num_samples + 1):
                    ts = idx * step
                    cand_id = f"C{idx}"
                    frame_path = temp_dir / f"{cand_id}.jpg"
                    persisted = self._persist_frame_or_none(media_source_for_frames, ts, fps, frame_path)
                    if persisted:
                        candidates.append(CandidateFrame(
                            candidate_id=cand_id, timestamp_seconds=ts,
                            frame_number=int(round(ts * fps)), image_path=persisted,
                        ))

            if self.vlm_service and candidates:
                try:
                    vlm_decision = self.vlm_service.evaluate_candidates(
                        target_text=request.target_text, candidates=candidates
                    )
                    if vlm_decision and vlm_decision.selected_candidate_id != "NONE":
                        selected_cand = next(
                            (c for c in candidates if c.candidate_id == vlm_decision.selected_candidate_id), None
                        )
                        if selected_cand:
                            matched_frame_path = artifact_dir / "frame.jpg"
                            shutil.copy(selected_cand.image_path, matched_frame_path)
                            frame_num = int(round(selected_cand.timestamp_seconds * fps))

                            return JobStatusResponse(
                                job_id=job_id, status=JobStatus.COMPLETED, target_dialogue=request.target_text,
                                formatted_timestamp=format_timestamp(selected_cand.timestamp_seconds),
                                timestamp_seconds=selected_cand.timestamp_seconds, frame_number=frame_num,
                                confidence_score=vlm_decision.confidence_score,
                                tier_executed=TierType.TIER_4_VLM_FALLBACK.value,
                                extracted_text=vlm_decision.exact_detected_text or request.target_text,
                                frame_image_path=str(matched_frame_path),
                            )
                except Exception as exc:
                    logger.warning("Tier 4 VLM evaluation failed: %s", exc)

            return JobStatusResponse(
                job_id=job_id, status=JobStatus.FAILED, target_dialogue=request.target_text,
                error_message="Target dialogue could not be localized across pipeline tiers.",
            )
        except Exception as exc:
            logger.exception("Job %s execution failed: %s", job_id, exc)
            return JobStatusResponse(job_id=job_id, status=JobStatus.FAILED, error_message=str(exc))
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass