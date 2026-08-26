import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("faster-whisper is required. Install it with: pip install faster-whisper")

from src.models.schemas import STTResult

logger = logging.getLogger(__name__)


class SpeechToTextService:
    def __init__(self, model_size: str = "turbo"):
        """Initializes local Whisper model. Downloads weights from HF once, then runs offline."""
        logger.info(f"Loading faster-whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="auto", compute_type="default")
        logger.info("Faster-whisper model loaded successfully.")

    def transcribe_audio(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Transcribes audio file and returns word-level timestamps."""
        logger.info(f"Transcribing audio: {audio_path}")

        segments, _ = self.model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en",
        )

        words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    clean_word = word.word.strip()
                    if clean_word:
                        words.append({
                            "word": clean_word,
                            "start": float(word.start),
                            "end": float(word.end),
                        })

        logger.info(f"Transcription complete. Found {len(words)} words.")
        return words

    def align_target_dialogue(self, words: List[Dict[str, Any]], target_text: str) -> STTResult:
        """Finds the best matching sequence of words in the transcription."""
        if not words or not target_text:
            return STTResult(found=False)

        target_lower = [t.lower().strip(".,!?;:") for t in target_text.split() if t.strip()]
        transcript_words = [w["word"].lower().strip(".,!?;:") for w in words]

        if not target_lower or len(transcript_words) < len(target_lower):
            return STTResult(found=False)

        best_match = None
        best_score = 0.0

        for i in range(len(transcript_words) - len(target_lower) + 1):
            window = transcript_words[i : i + len(target_lower)]
            matches = sum(1 for a, b in zip(window, target_lower) if a == b)
            score = matches / len(target_lower)

            if score > best_score:
                best_score = score
                best_match = i

        if best_match is not None and best_score >= 0.70:
            start_idx = best_match
            # Fixed off-by-one: grab the end timestamp of the LAST word in the matched sequence
            end_idx = min(best_match + len(target_lower) - 1, len(words) - 1)

            return STTResult(
                found=True,
                start_time=words[start_idx]["start"],
                end_time=words[end_idx]["end"],
                matched_text=target_text,
                confidence=round(best_score, 4),
            )

        return STTResult(found=False)