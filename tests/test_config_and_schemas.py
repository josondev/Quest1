import pytest
from pydantic import ValidationError
from src.config import Settings
from src.models.schemas import (
    JobRequest,
    BoundingBox,
    WordTimestamp,
    STTResult,
    OCRResult,
    CandidateFrame,
    VLMDecision,
    DetectionResult,
    JobStatus,
    TierType,
)


class TestConfig:
    def test_settings_initialization(self):
        """Verify that Settings initializes with default values and types."""
        settings = Settings()
        assert settings.confidence_threshold == 0.85
        assert settings.weight_similarity == 0.45
        assert settings.weight_ocr_confidence == 0.35
        assert settings.weight_temporal_alignment == 0.20
        assert settings.artifact_storage_dir.exists()
        assert settings.temp_storage_dir.exists()

    def test_weights_sum_to_one(self):
        """Verify that fusion weights sum to 1.0 (within floating point precision)."""
        settings = Settings()
        total_weight = (
            settings.weight_similarity
            + settings.weight_ocr_confidence
            + settings.weight_temporal_alignment
        )
        assert abs(total_weight - 1.0) < 1e-6


class TestSchemas:
    def test_job_request_valid(self):
        """Verify valid job request creation."""
        req = JobRequest(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            target_text="Never gonna give you up"
        )
        assert req.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert req.target_text == "Never gonna give you up"

    def test_job_request_invalid_url(self):
        """Verify that non-http/https URLs are rejected."""
        with pytest.raises(ValidationError):
            JobRequest(url="ftp://example.com/video.mp4", target_text="Hello")

        with pytest.raises(ValidationError):
            JobRequest(url="not_a_url", target_text="Hello")

    def test_job_request_empty_text(self):
        """Verify that empty or whitespace-only target text is rejected."""
        with pytest.raises(ValidationError):
            JobRequest(url="https://example.com/video.mp4", target_text="   ")

    def test_bounding_box_validation(self):
        """Verify bounding box boundary constraints [0.0, 1.0]."""
        bbox = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9)
        assert bbox.ymin == 0.1
        assert bbox.xmax == 0.9

        # Out-of-bounds check
        with pytest.raises(ValidationError):
            BoundingBox(ymin=-0.1)

        with pytest.raises(ValidationError):
            BoundingBox(ymax=1.5)

    def test_word_timestamp_schema(self):
        """Verify WordTimestamp data structure."""
        wt = WordTimestamp(word="stagnation", start=12.4, end=13.1, probability=0.98)
        assert wt.word == "stagnation"
        assert wt.start == 12.4
        assert wt.end == 13.1

    def test_stt_result_schema(self):
        """Verify STTResult with embedded word timestamps."""
        stt = STTResult(
            found=True,
            matched_text="My mind rebels at stagnation",
            start_time=10.5,
            end_time=13.2,
            confidence=0.94,
            words=[
                WordTimestamp(word="My", start=10.5, end=10.8),
                WordTimestamp(word="mind", start=10.8, end=11.2),
            ]
        )
        assert stt.found is True
        assert len(stt.words) == 2

    def test_candidate_frame_and_vlm_decision(self):
        """Verify candidate frame and VLM decision schemas."""
        candidate = CandidateFrame(
            candidate_id="C1",
            timestamp_seconds=12.5,
            frame_number=375,
            image_path="./artifacts/test_c1.jpg",
            ocr_detected_text="rebels at stagnation",
            ocr_confidence=0.78
        )
        assert candidate.candidate_id == "C1"

        decision = VLMDecision(
            selected_candidate_id="C1",
            exact_detected_text="My mind rebels at stagnation",
            confidence_score=0.96,
            reasoning="Dialogue visually legible and matches target phrase."
        )
        assert decision.selected_candidate_id == "C1"

    def test_detection_result_serialization(self):
        """Verify final DetectionResult serialization."""
        res = DetectionResult(
            job_id="job_test_123",
            status=JobStatus.COMPLETED,
            target_dialogue="My mind rebels at stagnation",
            timestamp_seconds=12.5,
            formatted_timestamp="00:00:12.500",
            frame_number=375,
            extracted_text="My mind rebels at stagnation",
            confidence_score=0.92,
            tier_executed=TierType.TIER_1_STT_DENSE,
            frame_image_path="./artifacts/frame_375.jpg",
            cropped_roi_path="./artifacts/roi_375.jpg"
        )
        data = res.model_dump()
        assert data["job_id"] == "job_test_123"
        assert data["status"] == "completed"
        assert data["tier_executed"] == "Tier 1: STT Acoustic Match + Dense OCR"
