"""
tests/test_primary_ocr.py
Unit tests for Step 4: Visual Frame Sampling + Mistral OCR

Tests cover:
    - frame_to_base64: JPEG encoding of a numpy frame
    - crop_subtitle_roi: subtitle zone cropping
    - _extract_text_from_response: Mistral API response parsing
    - ocr_frame: full OCR call with a mocked Mistral client
    - sample_frames_sparse: sparse frame sampling via mocked OpenCV
    - sample_frames_dense: dense frame sampling via mocked OpenCV
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.primary_ocr import (
    OCRError,
    _extract_text_from_response,
    crop_subtitle_roi,
    frame_to_base64,
    ocr_frame,
    sample_frames_dense,
    sample_frames_sparse,
)
from src.models.schemas import BoundingBox, OCRResult


# ---------------------------------------------------------------------------
# Helper: Build a blank test frame (green 640x360)
# ---------------------------------------------------------------------------

def make_test_frame(height: int = 360, width: int = 640) -> np.ndarray:
    """Create a blank BGR test frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test: frame_to_base64
# ---------------------------------------------------------------------------

class TestFrameToBase64:

    def test_returns_valid_base64_string(self):
        """Encoding a valid frame should return a non-empty base64 string."""
        frame = make_test_frame()
        result = frame_to_base64(frame)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_base64_is_decodable(self):
        """The base64 string should decode back to valid JPEG bytes."""
        frame = make_test_frame()
        b64 = frame_to_base64(frame)
        raw = base64.b64decode(b64)
        # JPEG magic bytes: FF D8 FF
        assert raw[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# Test: crop_subtitle_roi
# ---------------------------------------------------------------------------

class TestCropSubtitleROI:

    def test_returns_bottom_25_percent(self):
        """Default crop should return only the bottom 25% of the frame."""
        frame = make_test_frame(height=400, width=640)
        cropped = crop_subtitle_roi(frame, roi_fraction=0.25)
        assert cropped.shape[0] == 100   # 25% of 400
        assert cropped.shape[1] == 640

    def test_falls_back_to_full_frame_if_too_small(self):
        """If the crop would be < 32px tall, the full frame is returned."""
        # Height=100, roi_fraction=0.2 → crop height = 20px < 32px threshold
        frame = make_test_frame(height=100, width=640)
        result = crop_subtitle_roi(frame, roi_fraction=0.2)
        assert result.shape[0] == 100   # full frame

    def test_custom_roi_fraction(self):
        """Custom roi_fraction should scale correctly."""
        frame = make_test_frame(height=1080, width=1920)
        cropped = crop_subtitle_roi(frame, roi_fraction=0.3)
        assert cropped.shape[0] == 324   # 30% of 1080


# ---------------------------------------------------------------------------
# Test: _extract_text_from_response
# ---------------------------------------------------------------------------

class TestExtractTextFromResponse:

    def _make_mock_response(self, markdown: str, blocks=None):
        """Build a mock Mistral OCR response object."""
        page = MagicMock()
        page.markdown = markdown
        page.blocks = blocks or []
        response = MagicMock()
        response.pages = [page]
        return response

    def test_extracts_markdown_text(self):
        """Text should be extracted from response.pages[0].markdown."""
        response = self._make_mock_response("Hello World from subtitle")
        text, confidence, box = _extract_text_from_response(response)
        assert text == "Hello World from subtitle"

    def test_empty_pages_returns_empty(self):
        """A response with no pages should return empty string and zero confidence."""
        response = MagicMock()
        response.pages = []
        text, confidence, box = _extract_text_from_response(response)
        assert text == ""
        assert confidence == 0.0
        assert box is None

    def test_confidence_scales_with_text_length(self):
        """Confidence should scale with text length (longer = higher confidence up to 1.0)."""
        # Exactly 50 chars → confidence = 1.0
        response_long = self._make_mock_response("A" * 50)
        _, conf_long, _ = _extract_text_from_response(response_long)
        assert conf_long == 1.0

        # Empty text → confidence = 0.0
        response_empty = self._make_mock_response("")
        _, conf_empty, _ = _extract_text_from_response(response_empty)
        assert conf_empty == 0.0

    def test_extracts_bounding_box_from_first_block(self):
        """BoundingBox should be populated from the first block's coordinates."""
        block = MagicMock()
        block.top_left_x = 0.1
        block.top_left_y = 0.8
        block.bottom_right_x = 0.9
        block.bottom_right_y = 1.0
        response = self._make_mock_response("subtitle text", blocks=[block])
        _, _, box = _extract_text_from_response(response)
        assert box is not None
        assert box.xmin == pytest.approx(0.1)
        assert box.ymin == pytest.approx(0.8)
        assert box.xmax == pytest.approx(0.9)
        assert box.ymax == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: ocr_frame (mocked Mistral client)
# ---------------------------------------------------------------------------

class TestOCRFrame:

    def _make_client(self, markdown: str = "Elementary, my dear Watson"):
        """Build a mock Mistral client whose ocr.process returns markdown text."""
        client = MagicMock()
        page = MagicMock()
        page.markdown = markdown
        page.blocks = []
        response = MagicMock()
        response.pages = [page]
        client.ocr.process.return_value = response
        return client

    def test_returns_ocr_result_schema(self):
        """ocr_frame should return a valid OCRResult instance."""
        frame = make_test_frame()
        client = self._make_client()
        result = ocr_frame(frame, client, frame_index=42, timestamp_seconds=1.68)
        assert isinstance(result, OCRResult)

    def test_extracted_text_matches_mock(self):
        """The detected_text field should match what the mock OCR returns."""
        frame = make_test_frame()
        client = self._make_client("My mind rebels at stagnation")
        result = ocr_frame(frame, client, frame_index=10, timestamp_seconds=0.4)
        assert "stagnation" in result.detected_text.lower()

    def test_frame_index_and_timestamp_preserved(self):
        """frame_index and timestamp_seconds should be stored in the result."""
        frame = make_test_frame()
        client = self._make_client("some text")
        result = ocr_frame(frame, client, frame_index=99, timestamp_seconds=3.96)
        assert result.frame_index == 99
        assert result.timestamp_seconds == pytest.approx(3.96)

    def test_ocr_api_failure_raises_ocr_error(self):
        """When the Mistral API raises an exception, an OCRError should be raised."""
        frame = make_test_frame()
        client = MagicMock()
        client.ocr.process.side_effect = RuntimeError("API timeout")
        with pytest.raises(OCRError, match="Mistral OCR API call failed"):
            ocr_frame(frame, client, frame_index=0, timestamp_seconds=0.0)


# ---------------------------------------------------------------------------
# Test: sample_frames_sparse (mocked OpenCV VideoCapture)
# ---------------------------------------------------------------------------

class TestSampleFramesSparse:

    @patch("src.primary_ocr.cv2.VideoCapture")
    def test_samples_correct_number_of_frames(self, mock_cap_cls, tmp_path):
        """At 0.5 FPS with native 25 FPS, interval = 50 frames. 100 total frames → 2 samples."""
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 25.0 if prop == cv2.CAP_PROP_FPS else 100

        # Return a valid frame, then fail to stop the loop
        fake_frame = make_test_frame()
        mock_cap.read.side_effect = [
            (True, fake_frame),   # frame 0
            (True, fake_frame),   # frame 50
            (False, None),        # stop
        ]

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        frames = sample_frames_sparse(video_path, sample_fps=0.5)
        assert len(frames) == 2

    @patch("src.primary_ocr.cv2.VideoCapture")
    def test_raises_error_on_unreadable_file(self, mock_cap_cls, tmp_path):
        """Should raise OCRError if OpenCV cannot open the video file."""
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = False

        video_path = tmp_path / "bad_video.mp4"
        video_path.touch()

        with pytest.raises(OCRError, match="Cannot open video file"):
            sample_frames_sparse(video_path, sample_fps=0.5)
