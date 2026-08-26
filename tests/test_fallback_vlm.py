# tests/test_fallback_vlm.py
import pytest
from unittest.mock import MagicMock, patch
from src.fallback_vlm import VLMArbiterService, VLMError, encode_image_to_base64
from src.models.schemas import CandidateFrame, VLMDecision


class TestVLMArbiterService:

    def test_encode_image_to_base64_valid(self, tmp_path):
        test_file = tmp_path / "frame_001.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        b64, mime = encode_image_to_base64(str(test_file))
        assert mime == "image/jpeg"
        assert isinstance(b64, str)

    def test_encode_image_to_base64_invalid_path(self):
        with pytest.raises(VLMError, match="Candidate frame image not found"):
            encode_image_to_base64("non_existent_frame.jpg")

    def test_encode_image_to_base64_unsupported_extension(self, tmp_path):
        test_file = tmp_path / "frame_001.txt"
        test_file.write_bytes(b"not an image")
        with pytest.raises(VLMError, match="Unsupported or undetected image type"):
            encode_image_to_base64(str(test_file))

    def test_evaluate_candidates_empty_returns_none(self):
        service = VLMArbiterService(api_key="mock_key")
        result = service.evaluate_candidates("target phrase", [])
        assert result.selected_candidate_id == "NONE"
        assert result.confidence_score == 0.0

    def test_evaluate_candidates_requires_api_key(self, tmp_path):
        test_frame = tmp_path / "C1.jpg"
        test_frame.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        service = VLMArbiterService(api_key="")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=10.0, frame_number=240, image_path=str(test_frame))]

        with pytest.raises(VLMError, match="API Key is not configured"):
            service.evaluate_candidates("target phrase", candidates)

    def test_evaluate_candidates_rejects_duplicate_ids(self, tmp_path):
        test_frame = tmp_path / "C1.jpg"
        test_frame.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        service = VLMArbiterService(api_key="mock_key")
        candidates = [
            CandidateFrame(candidate_id="C1", timestamp_seconds=10.0, frame_number=240, image_path=str(test_frame)),
            CandidateFrame(candidate_id="C1", timestamp_seconds=12.0, frame_number=280, image_path=str(test_frame)),
        ]
        with pytest.raises(VLMError, match="Candidate IDs provided for VLM evaluation must be unique"):
            service.evaluate_candidates("target phrase", candidates)

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_full_schema_success(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f2 = tmp_path / "C2.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")
        f2.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "selected_candidate_id": "C2",
            "exact_detected_text": "My mind rebels at stagnation",
            "bounding_box": {"ymin": 0.75, "xmin": 0.1, "ymax": 0.95, "xmax": 0.9},
            "confidence_score": 0.95,
            "reasoning": "Subtitles clearly present in lower region."
        }"""
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [
            CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1)),
            CandidateFrame(candidate_id="C2", timestamp_seconds=10.0, frame_number=240, image_path=str(f2)),
        ]
        decision = service.evaluate_candidates("My mind rebels at stagnation", candidates)
        assert decision.selected_candidate_id == "C2"
        assert decision.exact_detected_text == "My mind rebels at stagnation"
        assert decision.bounding_box.ymin == 0.75
        assert decision.confidence_score == 0.95
        assert decision.reasoning == "Subtitles clearly present in lower region."

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_verifies_request_payload(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f2 = tmp_path / "C2.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")
        f2.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "selected_candidate_id": "C1",
            "exact_detected_text": "target phrase",
            "bounding_box": null,
            "confidence_score": 0.9,
            "reasoning": "Found."
        }"""
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [
            CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1)),
            CandidateFrame(candidate_id="C2", timestamp_seconds=10.0, frame_number=240, image_path=str(f2)),
        ]
        service.evaluate_candidates("target phrase", candidates)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        # Ensure both candidate IDs and target text are in payload
        assert 'Target Dialogue to locate: "target phrase"' in user_content[0]["text"]
        assert "\nCandidate ID: C1:" in user_content[1]["text"]
        assert "\nCandidate ID: C2:" in user_content[3]["text"]

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_rejects_unbounded_confidence(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "selected_candidate_id": "C1",
            "exact_detected_text": "target",
            "confidence_score": 17.0,
            "reasoning": "Too high."
        }"""
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1))]
        with pytest.raises(VLMError, match="invalid confidence_score: 17.0"):
            service.evaluate_candidates("target", candidates)

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_rejects_missing_required_fields(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "selected_candidate_id": "C1",
            "confidence_score": 0.95
        }"""
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1))]
        with pytest.raises(VLMError, match="missing required field 'exact_detected_text'"):
            service.evaluate_candidates("target", candidates)

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_accepts_none_selection(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "selected_candidate_id": "NONE",
            "exact_detected_text": "",
            "confidence_score": 0.1,
            "reasoning": "Not present."
        }"""
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1))]
        decision = service.evaluate_candidates("target phrase", candidates)
        assert decision.selected_candidate_id == "NONE"

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_rejects_hallucinated_id(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """{
            "selected_candidate_id": "C99",
            "exact_detected_text": "target",
            "confidence_score": 0.9,
            "reasoning": "Hallucination."
        }"""
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1))]
        with pytest.raises(VLMError, match="which was not among the candidates provided"):
            service.evaluate_candidates("target phrase", candidates)

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_rejects_malformed_json(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Invalid Non-JSON response"
        mock_client.chat.completions.create.return_value = mock_response

        service = VLMArbiterService(api_key="mock_key")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1))]
        with pytest.raises(VLMError, match="not valid JSON"):
            service.evaluate_candidates("target phrase", candidates)

    @patch("src.fallback_vlm.OpenAI")
    def test_evaluate_candidates_handles_provider_exception(self, mock_openai_cls, tmp_path):
        f1 = tmp_path / "C1.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API connection timeout")

        service = VLMArbiterService(api_key="mock_key")
        candidates = [CandidateFrame(candidate_id="C1", timestamp_seconds=5.0, frame_number=120, image_path=str(f1))]
        with pytest.raises(VLMError, match="VLM decision processing failed"):
            service.evaluate_candidates("target phrase", candidates)