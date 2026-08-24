import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.models.schemas import WordTimestamp, STTResult
from src.primary_stt import SpeechToTextService, normalize_text


class TestSpeechToTextService:
    def test_normalize_text(self):
        """Verify punctuation removal and lowercasing."""
        assert normalize_text("Hello, World!") == "hello world"
        assert normalize_text("  My mind rebels... at stagnation!  ") == "my mind rebels at stagnation"
        assert normalize_text("") == ""

    def test_align_target_dialogue_exact_match(self):
        """Verify alignment when target phrase matches spoken words exactly."""
        words = [
            WordTimestamp(word="I", start=0.0, end=0.3, probability=0.99),
            WordTimestamp(word="know", start=0.3, end=0.6, probability=0.98),
            WordTimestamp(word="My", start=1.0, end=1.3, probability=0.95),
            WordTimestamp(word="mind", start=1.3, end=1.8, probability=0.97),
            WordTimestamp(word="rebels", start=1.8, end=2.4, probability=0.96),
            WordTimestamp(word="at", start=2.4, end=2.6, probability=0.99),
            WordTimestamp(word="stagnation", start=2.6, end=3.4, probability=0.94),
            WordTimestamp(word="Watson", start=3.8, end=4.2, probability=0.90),
        ]

        svc = SpeechToTextService()
        result = svc.align_target_dialogue(words, "My mind rebels at stagnation")

        assert result.found is True
        assert result.start_time == 1.0
        assert result.end_time == 3.4
        assert result.confidence >= 0.85
        assert len(result.words) == 5

    def test_align_target_dialogue_fuzzy_match(self):
        """Verify alignment when target phrase has minor spelling or casing differences."""
        words = [
            WordTimestamp(word="my", start=5.0, end=5.3),
            WordTimestamp(word="mind", start=5.3, end=5.7),
            WordTimestamp(word="rebels", start=5.7, end=6.2),
            WordTimestamp(word="against", start=6.2, end=6.6),
            WordTimestamp(word="stagnation", start=6.6, end=7.3),
        ]

        svc = SpeechToTextService()
        result = svc.align_target_dialogue(words, "My mind rebels at stagnation", min_similarity_threshold=0.65)

        assert result.found is True
        assert result.start_time == 5.0
        assert result.end_time == 7.3

    def test_align_target_dialogue_not_found(self):
        """Verify found=False when dialogue does not exist in transcription."""
        words = [
            WordTimestamp(word="Elementary", start=1.0, end=1.5),
            WordTimestamp(word="my", start=1.5, end=1.8),
            WordTimestamp(word="dear", start=1.8, end=2.1),
            WordTimestamp(word="Watson", start=2.1, end=2.6),
        ]

        svc = SpeechToTextService()
        result = svc.align_target_dialogue(words, "To be or not to be that is the question")
        assert result.found is False

    @patch("src.primary_stt.Groq")
    def test_transcribe_with_groq_mock(self, mock_groq_cls, tmp_path):
        """Test Groq transcription with mocked word timestamps response."""
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance

        mock_transcription = MagicMock()
        mock_transcription.words = [
            {"word": "Never", "start": 0.1, "end": 0.4, "probability": 0.99},
            {"word": "gonna", "start": 0.4, "end": 0.7, "probability": 0.98},
            {"word": "give", "start": 0.7, "end": 1.0, "probability": 0.97},
            {"word": "you", "start": 1.0, "end": 1.2, "probability": 0.99},
            {"word": "up", "start": 1.2, "end": 1.5, "probability": 0.99},
        ]
        mock_groq_instance.audio.transcriptions.create.return_value = mock_transcription

        dummy_audio = tmp_path / "test.mp3"
        dummy_audio.write_bytes(b"dummy")

        svc = SpeechToTextService(provider="groq", groq_api_key="mock_key")
        words = svc.transcribe_audio(dummy_audio)

        assert len(words) == 5
        assert words[0].word == "Never"
        assert words[-1].end == 1.5
