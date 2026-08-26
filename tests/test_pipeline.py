import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.fallback_vlm import VLMError
from src.ingestion import IngestionError
from src.models.schemas import (
    CandidateFrame,
    DetectionResult,
    JobRequest,
    JobStatus,
    STTResult,
    SubtitleMatchResult,
    TierType,
    VLMDecision,
    VideoMetadata,
    WordTimestamp,
)
from src.pipeline import PipelineOrchestrator, calculate_confidence_score, format_timestamp


class TestPipelineOrchestrator:

    def test_format_timestamp(self):
        assert format_timestamp(0.0) == "00:00:00.000"
        assert format_timestamp(65.432) == "00:01:05.432"
        assert format_timestamp(3661.050) == "01:01:01.050"
        assert format_timestamp(-10.0) == "00:00:00.000"

    def test_format_timestamp_non_finite(self):
        assert format_timestamp(float("inf")) == "00:00:00.000"
        assert format_timestamp(float("-inf")) == "00:00:00.000"
        assert format_timestamp(float("nan")) == "00:00:00.000"

    def test_calculate_confidence_score(self):
        score = calculate_confidence_score(text_similarity=0.9, ocr_confidence=0.8, alignment_score=1.0)
        assert score == round(0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 1.0, 4)

    @patch("src.pipeline.StreamIngestionService")
    def test_run_tier0_success_short_circuits(self, mock_ingestion_cls):
        mock_ingestion = MagicMock()
        mock_ingestion.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=True,
            is_local=False,
        )
        mock_ingestion.probe_embedded_subtitles_match.return_value = SubtitleMatchResult(
            start_time=12.5,
            end_time=15.0,
            matched_text="My mind rebels at stagnation",
            similarity_score=0.95,
            track_language="en",
        )

        orchestrator = PipelineOrchestrator(ingestion_service=mock_ingestion)
        request = JobRequest(url="https://youtube.com/watch?v=test", target_text="My mind rebels at stagnation")
        result = orchestrator.run(job_id="test_001", request=request)

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_0_SUBTITLE
        assert result.timestamp_seconds == 12.5
        assert result.frame_number == 312
        mock_ingestion.extract_audio_stream.assert_not_called()

    @patch("src.pipeline.scan_sparse_timeline")
    @patch("src.pipeline.scan_dense_window")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_stt_fallback_order_tier1_to_tier3_to_tier2_to_tier1(
        self, mock_ingestion, mock_stt, mock_dense_ocr, mock_sparse_ocr, tmp_path
    ):
        mock_ingestion.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            stream_path="https://googlevideo.com/videoplayback_direct_stream",
        )
        mock_ingestion.extract_audio_stream.return_value = tmp_path / "audio.wav"

        mock_stt.transcribe_audio.return_value = [WordTimestamp(word="hello", start=10.0, end=10.5)]
        mock_stt.align_target_dialogue.return_value = STTResult(
            found=True, matched_text="hello world", start_time=10.0, end_time=10.5, confidence=0.88
        )

        mock_dense_ocr.return_value = None
        mock_sparse_ocr.return_value = None

        mock_vlm = MagicMock()
        mock_vlm.evaluate_candidates.return_value = VLMDecision(selected_candidate_id="NONE")

        orchestrator = PipelineOrchestrator(
            ingestion_service=mock_ingestion,
            stt_service=mock_stt,
            vlm_service=mock_vlm,
            mistral_client=MagicMock(),
        )
        request = JobRequest(url="https://youtube.com/watch?v=test", target_text="hello world")
        result = orchestrator.run(job_id="test_fallback_order", request=request)

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_1_STT
        assert result.timestamp_seconds == 10.0

    @patch("src.pipeline.scan_dense_window")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_no_fabricated_ocr_confidence(self, mock_ingestion, mock_stt, mock_dense_ocr, tmp_path):
        mock_ingestion.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            stream_path="https://googlevideo.com/videoplayback_direct_stream",
        )
        mock_ingestion.extract_audio_stream.return_value = tmp_path / "audio.wav"

        mock_stt.transcribe_audio.return_value = [WordTimestamp(word="test", start=5.0, end=5.5)]
        mock_stt.align_target_dialogue.return_value = STTResult(
            found=True, matched_text="test dialogue", start_time=5.0, end_time=5.5, confidence=0.88
        )

        dense_cand = MagicMock()
        dense_cand.timestamp_seconds = 5.2
        dense_cand.frame_number = 130
        dense_cand.ocr_detected_text = "test dialogue"
        dense_cand.ocr_confidence = 0.65
        mock_dense_ocr.return_value = dense_cand

        orchestrator = PipelineOrchestrator(
            ingestion_service=mock_ingestion,
            stt_service=mock_stt,
            mistral_client=MagicMock(),
        )
        request = JobRequest(url="https://youtube.com/watch?v=test", target_text="test dialogue")
        result = orchestrator.run(job_id="test_no_fab_ocr", request=request)

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_3_DENSE_OCR

        expected_score = round(0.5 * 0.88 + 0.3 * 0.65 + 0.2 * 1.0, 4)
        assert result.confidence_score == expected_score

    @patch("src.pipeline.cv2.imwrite")
    @patch("src.pipeline.cv2.VideoCapture")
    @patch("src.pipeline.VLMArbiterService")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_persistent_vlm_frame_artifact_survives_temp_cleanup(
        self, mock_ingestion, mock_stt, mock_vlm, mock_cap_cls, mock_imwrite, tmp_path
    ):
        mock_ingestion.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            stream_path="https://googlevideo.com/videoplayback_direct_stream",
        )
        mock_ingestion.extract_audio_stream.return_value = tmp_path / "audio.wav"

        mock_stt.transcribe_audio.return_value = []
        mock_stt.align_target_dialogue.return_value = STTResult(found=False)

        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((360, 640, 3), dtype=np.uint8))

        def fake_imwrite(path, frame):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"JPEGDATA")
            return True

        mock_imwrite.side_effect = fake_imwrite

        mock_vlm.evaluate_candidates.return_value = VLMDecision(
            selected_candidate_id="C1",
            exact_detected_text="target dialogue",
            confidence_score=0.96,
        )

        orchestrator = PipelineOrchestrator(
            ingestion_service=mock_ingestion,
            stt_service=mock_stt,
            vlm_service=mock_vlm,
        )
        request = JobRequest(url="https://youtube.com/watch?v=test", target_text="target dialogue")
        job_id = "test_persist_vlm"
        result = orchestrator.run(job_id=job_id, request=request)

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_4_VLM_FALLBACK
        assert result.frame_image_path is not None

        persisted_path = Path(result.frame_image_path)
        assert persisted_path.exists()
        assert "quest1_test_persist_vlm_" not in str(persisted_path)
        assert persisted_path.read_bytes() == b"JPEGDATA"

        shutil.rmtree(persisted_path.parent, ignore_errors=True)

    @patch("src.pipeline.scan_dense_window")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_remote_media_source_passed_to_videocapture(self, mock_ingestion, mock_stt, mock_dense_ocr, tmp_path):
        remote_stream_url = "https://googlevideo.com/videoplayback_direct_stream"

        mock_ingestion.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            stream_path=remote_stream_url,
        )
        mock_ingestion.extract_audio_stream.return_value = tmp_path / "audio.wav"

        mock_stt.transcribe_audio.return_value = [WordTimestamp(word="test", start=5.0, end=5.5)]
        mock_stt.align_target_dialogue.return_value = STTResult(
            found=True, matched_text="test dialogue", start_time=5.0, end_time=5.5, confidence=0.88
        )

        dense_cand = MagicMock()
        dense_cand.timestamp_seconds = 5.2
        dense_cand.frame_number = 130
        dense_cand.ocr_detected_text = "test dialogue"
        dense_cand.ocr_confidence = 0.90
        mock_dense_ocr.return_value = dense_cand

        orchestrator = PipelineOrchestrator(
            ingestion_service=mock_ingestion,
            stt_service=mock_stt,
            mistral_client=MagicMock(),
        )
        request = JobRequest(url="https://youtube.com/watch?v=test", target_text="test dialogue")
        orchestrator.run(job_id="test_remote_source", request=request)

        mock_dense_ocr.assert_called_once()
        passed_source = mock_dense_ocr.call_args[0][0]
        assert passed_source == remote_stream_url

    @patch("src.pipeline.StreamIngestionService")
    @patch("src.pipeline.shutil.rmtree")
    def test_temp_directory_cleaned_up_on_completion(self, mock_rmtree, mock_ingestion_cls):
        mock_ingestion = MagicMock()
        mock_ingestion.probe_metadata.side_effect = Exception("General error")

        orchestrator = PipelineOrchestrator(ingestion_service=mock_ingestion)
        request = JobRequest(url="https://youtube.com/watch?v=test", target_text="target")
        orchestrator.run(job_id="test_cleanup", request=request)

        assert mock_rmtree.called