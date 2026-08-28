import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rapidfuzz import fuzz

from src.config import settings
from src.models.schemas import STTResult, WordTimestamp

# Module-level Groq object for @patch("src.primary_stt.Groq")
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SpeechToTextService:
    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider or settings.whisper_provider
        self.groq_api_key = groq_api_key or settings.groq_api_key
        self.hf_token = hf_token or settings.hf_token
        self.model_name = model_name or settings.whisper_model_name

        self._groq_client: Optional[Any] = None
        self._hf_client: Optional[Any] = None

        if self.provider not in {"groq", "huggingface"}:
            raise ValueError(
                f"Unsupported Whisper STT provider: {self.provider}. "
                "Expected 'groq' or 'huggingface'."
            )

    def _get_groq_client(self) -> Any:
        if self._groq_client is not None:
            return self._groq_client

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        if Groq is None:
            raise ImportError("groq package is required for Groq STT.")

        self._groq_client = Groq(
            api_key=self.groq_api_key,
            timeout=60.0,
        )
        return self._groq_client

    def _get_hf_client(self) -> Any:
        if self._hf_client is not None:
            return self._hf_client

        if InferenceClient is None:
            raise ImportError("huggingface_hub is required for Hugging Face STT.")

        self._hf_client = InferenceClient(
            token=self.hf_token if self.hf_token else None
        )
        return self._hf_client

    @staticmethod
    def _validate_audio_file(audio_path: Path) -> None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")
        if audio_path.stat().st_size <= 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

    def transcribe_audio(
        self,
        audio_path: Path,
    ) -> List[WordTimestamp]:
        audio_path = Path(audio_path)
        self._validate_audio_file(audio_path)

        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        if file_size_mb < 10.0:
            if self.provider == "groq":
                return self._transcribe_with_groq(audio_path, offset=0.0)
            if self.provider == "huggingface":
                return self._transcribe_with_hf(audio_path, offset=0.0)
            raise ValueError(f"Unsupported provider: {self.provider}")

        words: List[WordTimestamp] = []
        with tempfile.TemporaryDirectory(prefix="stt_chunks_") as tmpdir:
            chunk_pattern = Path(tmpdir) / "chunk_%03d.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-f", "segment", "-segment_time", "180",
                "-c", "copy", str(chunk_pattern)
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"FFmpeg chunking failed: {exc}") from exc

            chunks = sorted(Path(tmpdir).glob("chunk_*.wav"))
            for idx, chunk_path in enumerate(chunks):
                offset_seconds = idx * 180.0
                if self.provider == "groq":
                    chunk_words = self._transcribe_with_groq(chunk_path, offset=offset_seconds)
                elif self.provider == "huggingface":
                    chunk_words = self._transcribe_with_hf(chunk_path, offset=offset_seconds)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                words.extend(chunk_words)

        return words

    def _transcribe_with_groq(
        self,
        audio_path: Path,
        offset: float = 0.0,
    ) -> List[WordTimestamp]:
        client = self._get_groq_client()
        max_attempts = 3
        transcription = None

        for attempt in range(max_attempts):
            try:
                with audio_path.open("rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        file=(audio_path.name, audio_file.read()),
                        model=self.model_name,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        temperature=0.0,
                    )
                break
            except Exception as exc:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Groq STT failed: {exc}") from exc
                time.sleep(2 ** attempt)

        words: List[WordTimestamp] = []
        raw_words = getattr(transcription, "words", None) or []

        for raw_word in raw_words:
            try:
                if isinstance(raw_word, dict):
                    word_text = str(raw_word.get("word", "")).strip()
                    raw_start = float(raw_word.get("start", 0.0) or 0.0)
                    raw_end = float(raw_word.get("end", raw_start) or raw_start)
                    probability_raw = raw_word.get("probability", raw_word.get("avg_logprob", 1.0))
                else:
                    word_text = str(getattr(raw_word, "word", "")).strip()
                    raw_start = float(getattr(raw_word, "start", 0.0) or 0.0)
                    raw_end = float(getattr(raw_word, "end", raw_start) or raw_start)
                    probability_raw = getattr(raw_word, "probability", 1.0)

                try:
                    probability = float(probability_raw)
                except (TypeError, ValueError):
                    probability = 1.0

                probability = max(0.0, min(1.0, probability))

                if word_text:
                    words.append(
                        WordTimestamp(
                            word=word_text,
                            start=raw_start + offset,
                            end=raw_end + offset,
                            probability=probability,
                        )
                    )
            except Exception:
                pass

        return words

    def _transcribe_with_hf(
        self,
        audio_path: Path,
        offset: float = 0.0,
    ) -> List[WordTimestamp]:
        client = self._get_hf_client()
        try:
            with audio_path.open("rb") as audio_file:
                result = client.automatic_speech_recognition(
                    audio=audio_file.read(),
                    model=self.model_name,
                )
        except Exception as exc:
            raise RuntimeError(f"Hugging Face STT failed: {exc}") from exc

        chunks = result.get("chunks", []) if isinstance(result, dict) else (getattr(result, "chunks", None) or [])
        words: List[WordTimestamp] = []

        for chunk in chunks:
            try:
                if isinstance(chunk, dict):
                    text = str(chunk.get("text", "")).strip()
                    timestamp = chunk.get("timestamp", (0.0, 0.0))
                else:
                    text = str(getattr(chunk, "text", "")).strip()
                    timestamp = getattr(chunk, "timestamp", (0.0, 0.0))

                if not text:
                    continue

                if isinstance(timestamp, (tuple, list)) and len(timestamp) >= 2:
                    raw_start = float(timestamp[0] if timestamp[0] is not None else 0.0)
                    raw_end = float(timestamp[1] if timestamp[1] is not None else raw_start)
                    start = raw_start + offset
                    end = raw_end + offset
                else:
                    start = offset
                    end = offset

                words.append(
                    WordTimestamp(
                        word=text,
                        start=start,
                        end=end,
                        probability=0.9,
                    )
                )
            except Exception:
                pass

        return words

    def align_target_dialogue(
        self,
        words: List[WordTimestamp],
        target_dialogue: str,
        min_similarity_threshold: float = 0.70,
    ) -> STTResult:
        if not words or not target_dialogue or not target_dialogue.strip():
            return STTResult(found=False)

        normalized_target = normalize_text(target_dialogue)
        if not normalized_target:
            return STTResult(found=False)

        target_tokens = normalized_target.split()
        target_count = len(target_tokens)
        total_words = len(words)

        if total_words < target_count:
            return STTResult(found=False)

        best_score = 0.0
        best_window: Optional[Tuple[int, int]] = None

        min_window = max(1, target_count - 2)
        max_window = min(total_words, target_count + 3)

        for window_size in range(min_window, max_window + 1):
            for start_idx in range(0, total_words - window_size + 1):
                end_idx = start_idx + window_size
                window_words = words[start_idx:end_idx]

                candidate_text = " ".join(word.word for word in window_words)
                candidate_norm = normalize_text(candidate_text)

                if not candidate_norm:
                    continue

                score = fuzz.ratio(normalized_target, candidate_norm) / 100.0
                if score > best_score:
                    best_score = score
                    best_window = (start_idx, end_idx)

        if best_window is None or best_score < min_similarity_threshold:
            return STTResult(found=False, confidence=round(best_score, 4))

        start_idx, end_idx = best_window
        matched_words = words[start_idx:end_idx]

        start_time = float(matched_words[0].start)
        end_time = float(matched_words[-1].end)

        probabilities = [max(0.0, min(1.0, float(getattr(w, "probability", 1.0)))) for w in matched_words]
        mean_probability = sum(probabilities) / len(probabilities) if probabilities else 1.0

        composite_confidence = best_score * 0.70 + mean_probability * 0.30
        matched_text = " ".join(word.word for word in matched_words)

        return STTResult(
            found=True,
            matched_text=matched_text,
            start_time=start_time,
            end_time=end_time,
            confidence=round(composite_confidence, 4),
            words=matched_words,
        )