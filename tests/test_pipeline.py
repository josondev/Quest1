from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil
import wave
from src.ingestion import IngestionError

import numpy as np

from src.models.schemas import (
    JobRequest,
    JobStatus,
    STTResult,
    SubtitleMatchResult,
    TierType,
    VideoMetadata,
    VLMDecision,
    WordTimestamp,
)
from src.pipeline import (
    PipelineOrchestrator,
    calculate_confidence_score,
    format_timestamp,
)


def create_fake_wav(file_path: Path) -> Path:
    """
    Create a real, structurally valid PCM WAV file for unit tests.
    The audio contains silence because the STT service is mocked.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    channels = 1
    sample_width = 2
    duration_seconds = 1
    total_frames = sample_rate * duration_seconds

    silence_data = b"\x00\x00" * total_frames

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence_data)

    return file_path


class TestPipelineOrchestrator:

    # ============================================================
    # BASIC UTILITY TESTS
    # ============================================================

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
        text_similarity = 0.90
        ocr_confidence = 0.80
        alignment_score = 1.00

        expected_score = round(
            0.50 * text_similarity
            + 0.30 * ocr_confidence
            + 0.20 * alignment_score,
            4,
        )

        actual_score = calculate_confidence_score(
            text_similarity=text_similarity,
            ocr_confidence=ocr_confidence,
            alignment_score=alignment_score,
        )

        assert actual_score == expected_score

    def test_calculate_confidence_score_clamped(self):
        high_score = calculate_confidence_score(
            text_similarity=2.0,
            ocr_confidence=2.0,
            alignment_score=2.0,
        )

        low_score = calculate_confidence_score(
            text_similarity=-1.0,
            ocr_confidence=-1.0,
            alignment_score=-1.0,
        )

        assert high_score == 1.0
        assert low_score == 0.0

    # ============================================================
    # TIER 0
    # ============================================================

    @patch("src.pipeline.StreamIngestionService")
    def test_run_tier0_success_short_circuits(
        self,
        mock_ingestion_class,
    ):
        ingestion_service = MagicMock()

        ingestion_service.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=True,
            is_local=False,
            stream_path=(
                "https://googlevideo.com/"
                "videoplayback_direct_stream"
            ),
        )

        ingestion_service.probe_embedded_subtitles_match.return_value = (
            SubtitleMatchResult(
                start_time=12.5,
                end_time=15.0,
                matched_text="My mind rebels at stagnation",
                similarity_score=0.95,
                track_language="en",
                is_auto_generated=False,
            )
        )

        orchestrator = PipelineOrchestrator(
            ingestion_service=ingestion_service
        )

        request = JobRequest(
            url="https://youtube.com/watch?v=test",
            target_text="My mind rebels at stagnation",
        )

        result = orchestrator.run(
            job_id="test_tier0",
            request=request,
        )

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_0_SUBTITLE
        assert result.timestamp_seconds == 12.5
        assert result.frame_number == 312

        ingestion_service.extract_audio_stream.assert_not_called()

    # ============================================================
    # TIER 1 -> TIER 3 -> FALLBACK
    # ============================================================

    @patch("src.pipeline.scan_sparse_timeline")
    @patch("src.pipeline.scan_dense_window")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_stt_match_returns_tier1_result(
        self,
        mock_ingestion_class,
        mock_stt_class,
        mock_dense_ocr,
        mock_sparse_ocr,
        tmp_path,
    ):
        ingestion_service = MagicMock()
        stt_service = MagicMock()

        ingestion_service.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            is_local=False,
            stream_path=(
                "https://googlevideo.com/"
                "videoplayback_direct_stream"
            ),
        )

        audio_file = create_fake_wav(
            tmp_path / "audio.wav"
        )

        ingestion_service.extract_audio_stream.return_value = audio_file

        stt_service.transcribe_audio.return_value = [
            WordTimestamp(
                word="hello",
                start=10.0,
                end=10.5,
                probability=0.95,
            )
        ]

        stt_service.align_target_dialogue.return_value = STTResult(
            found=True,
            matched_text="hello world",
            start_time=10.0,
            end_time=10.5,
            confidence=0.88,
        )

        mock_dense_ocr.return_value = None
        mock_sparse_ocr.return_value = None

        vlm_service = MagicMock()

        vlm_service.evaluate_candidates.return_value = VLMDecision(
            selected_candidate_id="NONE",
            exact_detected_text="",
            confidence_score=0.0,
        )

        orchestrator = PipelineOrchestrator(
            ingestion_service=ingestion_service,
            stt_service=stt_service,
            vlm_service=vlm_service,
            mistral_client=MagicMock(),
        )

        request = JobRequest(
            url="https://youtube.com/watch?v=test",
            target_text="hello world",
        )

        result = orchestrator.run(
            job_id="test_tier1",
            request=request,
        )

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_1_STT
        assert result.timestamp_seconds == 10.0
        assert result.formatted_timestamp == "00:00:10.000"
        assert result.extracted_text == "hello world"
        assert result.confidence_score == 0.88

        ingestion_service.extract_audio_stream.assert_called_once()
        stt_service.transcribe_audio.assert_called_once_with(
            audio_file
        )
        stt_service.align_target_dialogue.assert_called_once()

    # ============================================================
    # TIER 3 OCR CONFIDENCE
    # ============================================================

    @patch("src.pipeline.scan_dense_window")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_tier3_uses_actual_ocr_confidence(
        self,
        mock_ingestion_class,
        mock_stt_class,
        mock_dense_ocr,
        tmp_path,
    ):
        ingestion_service = MagicMock()
        stt_service = MagicMock()

        ingestion_service.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            is_local=False,
            stream_path=(
                "https://googlevideo.com/"
                "videoplayback_direct_stream"
            ),
        )

        audio_file = create_fake_wav(
            tmp_path / "audio.wav"
        )

        ingestion_service.extract_audio_stream.return_value = audio_file

        stt_service.transcribe_audio.return_value = [
            WordTimestamp(
                word="test",
                start=5.0,
                end=5.5,
                probability=0.95,
            )
        ]

        stt_service.align_target_dialogue.return_value = STTResult(
            found=True,
            matched_text="test dialogue",
            start_time=5.0,
            end_time=5.5,
            confidence=0.88,
        )

        dense_candidate = MagicMock()

        dense_candidate.timestamp_seconds = 5.2
        dense_candidate.frame_number = 130
        dense_candidate.ocr_detected_text = "test dialogue"
        dense_candidate.ocr_confidence = 0.65
        dense_candidate.image_path = str(
            tmp_path / "tier3_verified.jpg"
        )

        mock_dense_ocr.return_value = dense_candidate

        orchestrator = PipelineOrchestrator(
            ingestion_service=ingestion_service,
            stt_service=stt_service,
            mistral_client=MagicMock(),
        )

        request = JobRequest(
            url="https://youtube.com/watch?v=test",
            target_text="test dialogue",
        )

        result = orchestrator.run(
            job_id="test_tier3",
            request=request,
        )

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_3_DENSE_OCR
        assert result.timestamp_seconds == 5.2
        assert result.extracted_text == "test dialogue"

        stt_confidence = 0.88
        ocr_confidence = 0.65
        alignment_score = 1.0

        expected_confidence = round(
            0.50 * stt_confidence
            + 0.30 * ocr_confidence
            + 0.20 * alignment_score,
            4,
        )

        assert result.confidence_score == expected_confidence

    # ============================================================
    # TIER 4 VLM ARTIFACT PERSISTENCE
    # ============================================================

    @patch("src.pipeline.cv2.imwrite")
    @patch("src.pipeline.cv2.VideoCapture")
    @patch("src.pipeline.VLMArbiterService")
    @patch("src.pipeline.SpeechToTextService")
    @patch("src.pipeline.StreamIngestionService")
    def test_vlm_frame_artifact_survives_temp_cleanup(
        self,
        mock_ingestion_class,
        mock_stt_class,
        mock_vlm_class,
        mock_video_capture,
        mock_imwrite,
        tmp_path,
    ):
        ingestion_service = MagicMock()
        stt_service = MagicMock()
        vlm_service = MagicMock()

        ingestion_service.probe_metadata.return_value = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            duration_seconds=100.0,
            fps=25.0,
            has_subtitles=False,
            is_local=False,
            stream_path=(
                "https://googlevideo.com/"
                "videoplayback_direct_stream"
            ),
        )

        audio_file = create_fake_wav(
            tmp_path / "audio.wav"
        )

        ingestion_service.extract_audio_stream.return_value = audio_file
        ingestion_service.extract_frame_on_demand.return_value = False

        stt_service.transcribe_audio.return_value = []
        stt_service.align_target_dialogue.return_value = (
            STTResult(found=False)
        )

        video_capture = MagicMock()

        video_capture.isOpened.return_value = True
        video_capture.read.return_value = (
            True,
            np.zeros(
                (360, 640, 3),
                dtype=np.uint8,
            ),
        )

        mock_video_capture.return_value = video_capture

        def fake_imwrite(
            output_path,
            frame,
        ):
            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_bytes(
                b"JPEGDATA"
            )

            return True

        mock_imwrite.side_effect = fake_imwrite

        vlm_service.evaluate_candidates.return_value = VLMDecision(
            selected_candidate_id="C1",
            exact_detected_text="target dialogue",
            confidence_score=0.96,
            reasoning="Matched candidate.",
        )

        orchestrator = PipelineOrchestrator(
            ingestion_service=ingestion_service,
            stt_service=stt_service,
            vlm_service=vlm_service,
        )

        request = JobRequest(
            url="https://youtube.com/watch?v=test",
            target_text="target dialogue",
        )

        job_id = "test_vlm_persistence"

        result = orchestrator.run(
            job_id=job_id,
            request=request,
        )

        assert result.status == JobStatus.COMPLETED
        assert result.tier_executed == TierType.TIER_4_VLM_FALLBACK
        assert result.frame_image_path is not None

        persisted_frame = Path(
            result.frame_image_path
        )

        assert persisted_frame.exists()

        assert (
            f"quest1_{job_id}_"
            not in str(persisted_frame)
        )

        assert persisted_frame.read_bytes() == b"JPEGDATA"

        shutil.rmtree(
            persisted_frame.parent,
            ignore_errors=True,
        )

    # ============================================================
    # TEMP DIRECTORY CLEANUP
    # ============================================================

    @patch("src.pipeline.StreamIngestionService")
    @patch("src.pipeline.shutil.rmtree")
    def test_temp_directory_cleaned_up(
        self,
        mock_rmtree,
        mock_ingestion_class,
    ):
        ingestion_service = MagicMock()

        ingestion_service.probe_metadata.side_effect = (
            Exception("General error")
        )

        orchestrator = PipelineOrchestrator(
            ingestion_service=ingestion_service
        )

        request = JobRequest(
            url="https://youtube.com/watch?v=test",
            target_text="target dialogue",
        )

        result = orchestrator.run(
            job_id="test_cleanup",
            request=request,
        )

        assert result.status == JobStatus.FAILED
        assert result.error_message is not None
        assert mock_rmtree.called

    # ============================================================
    # INGESTION ERROR HANDLING
    # ============================================================

    @patch("src.pipeline.StreamIngestionService")
    def test_ingestion_error_returns_failed_result(
        self,
        mock_ingestion_class,
    ):
        ingestion_service = MagicMock()

        ingestion_service.probe_metadata.side_effect = (
            IngestionError("Unable to open stream")
        )

        orchestrator = PipelineOrchestrator(
            ingestion_service=ingestion_service
        )

        request = JobRequest(
            url="https://ok.ru/video/test",
            target_text="My mind rebels at stagnation",
        )

        result = orchestrator.run(
            job_id="test_ingestion_failure",
            request=request,
        )

        assert result.status == JobStatus.FAILED
        assert result.error_message is not None
        assert (
            "Ingestion failed"
            in result.error_message
        )

    # ============================================================
    # WAV VALIDATION FIXTURE
    # ============================================================

    def test_create_fake_wav_creates_valid_wav(
        self,
        tmp_path,
    ):
        wav_file = create_fake_wav(
            tmp_path / "valid_audio.wav"
        )

        assert wav_file.exists()
        assert wav_file.is_file()
        assert wav_file.suffix.lower() == ".wav"
        assert wav_file.stat().st_size > 44

        with wave.open(
            str(wav_file),
            "rb",
        ) as wav_reader:
            assert wav_reader.getnchannels() == 1
            assert wav_reader.getsampwidth() == 2
            assert wav_reader.getframerate() == 16000
            assert wav_reader.getnframes() == 16000