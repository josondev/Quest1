import re
from pathlib import Path
from typing import List, Optional, Tuple
from groq import Groq
from huggingface_hub import InferenceClient
from rapidfuzz import fuzz

from src.config import settings
from src.models.schemas import STTResult, WordTimestamp


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for robust acoustic & visual fuzzy comparison."""
    if not text:
        return ""
    lowered = text.lower()
    # Replace punctuation with space and normalize whitespace
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(cleaned.split())


class SpeechToTextService:
    """
    Primary Acoustic Speech-to-Text (STT) Service.
    Transcribes audio with word-level timestamps and aligns spoken audio
    with the target dialogue using sliding window fuzzy string matching.
    """

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

        self._groq_client: Optional[Groq] = None
        self._hf_client: Optional[InferenceClient] = None

    def _get_groq_client(self) -> Groq:
        if not self._groq_client:
            if not self.groq_api_key:
                raise ValueError("Groq API Key is not configured. Please set GROQ_API_KEY in .env")
            self._groq_client = Groq(api_key=self.groq_api_key)
        return self._groq_client

    def _get_hf_client(self) -> InferenceClient:
        if not self._hf_client:
            self._hf_client = InferenceClient(token=self.hf_token if self.hf_token else None)
        return self._hf_client

    def transcribe_audio(self, audio_file_path: Path) -> List[WordTimestamp]:
        """
        Transcribe local audio file and return word-level timestamps.
        """
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file does not exist at {audio_file_path}")

        if self.provider == "groq":
            return self._transcribe_with_groq(audio_file_path)
        elif self.provider == "huggingface":
            return self._transcribe_with_hf(audio_file_path)
        else:
            raise ValueError(f"Unsupported Whisper STT provider: {self.provider}")

    def _transcribe_with_groq(self, audio_file_path: Path) -> List[WordTimestamp]:
        """Call Groq Whisper Large v3 with verbose_json for word timestamps."""
        client = self._get_groq_client()

        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file_path.name, file.read()),
                model=self.model_name,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                temperature=0.0,
            )

        words_list: List[WordTimestamp] = []
        raw_words = getattr(transcription, "words", None) or []
        for w in raw_words:
            # Handle both dict and object structures
            word_text = w.get("word") if isinstance(w, dict) else getattr(w, "word", "")
            start = float(w.get("start", 0.0) if isinstance(w, dict) else getattr(w, "start", 0.0))
            end = float(w.get("end", 0.0) if isinstance(w, dict) else getattr(w, "end", 0.0))
            prob = float(w.get("probability", 1.0) if isinstance(w, dict) else getattr(w, "probability", 1.0))

            clean_word = word_text.strip()
            if clean_word:
                words_list.append(
                    WordTimestamp(
                        word=clean_word,
                        start=start,
                        end=end,
                        probability=prob,
                    )
                )

        return words_list

    def _transcribe_with_hf(self, audio_file_path: Path) -> List[WordTimestamp]:
        """Fallback transcription via Hugging Face Inference API."""
        client = self._get_hf_client()
        with open(audio_file_path, "rb") as file:
            result = client.automatic_speech_recognition(
                audio=file.read(),
                model="openai/whisper-large-v3",
            )

        words_list: List[WordTimestamp] = []
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp", (0.0, 0.0))
            start, end = (timestamp[0] or 0.0, timestamp[1] or 0.0)
            if text:
                words_list.append(
                    WordTimestamp(
                        word=text,
                        start=start,
                        end=end,
                        probability=0.9,
                    )
                )

        return words_list

    def align_target_dialogue(
        self,
        words: List[WordTimestamp],
        target_dialogue: str,
        min_similarity_threshold: float = 0.70,
    ) -> STTResult:
        """
        Sliding window fuzzy alignment matching target dialogue against word timestamps.
        Finds the exact acoustic interval [t_start, t_end] with highest confidence.
        """
        if not words or not target_dialogue.strip():
            return STTResult(found=False)

        norm_target = normalize_text(target_dialogue)
        target_token_count = len(norm_target.split())
        if target_token_count == 0:
            return STTResult(found=False)

        # Sliding window range: target_token_count +/- 30%
        min_window = max(1, target_token_count - 2)
        max_window = target_token_count + 3

        best_score = 0.0
        best_window: Optional[Tuple[int, int]] = None
        best_text = ""

        total_words = len(words)
        for window_size in range(min_window, max_window + 1):
            if window_size > total_words:
                continue
            for i in range(0, total_words - window_size + 1):
                window_words = words[i : i + window_size]
                candidate_text = " ".join(w.word for w in window_words)
                norm_candidate = normalize_text(candidate_text)

                score = fuzz.token_sort_ratio(norm_target, norm_candidate) / 100.0

                if score > best_score:
                    best_score = score
                    best_window = (i, i + window_size)
                    best_text = candidate_text

        if best_window and best_score >= min_similarity_threshold:
            start_idx, end_idx = best_window
            matched_words = words[start_idx:end_idx]
            start_time = matched_words[0].start
            end_time = matched_words[-1].end

            mean_prob = sum(w.probability for w in matched_words) / len(matched_words)
            composite_stt_confidence = float(best_score * 0.7 + mean_prob * 0.3)

            return STTResult(
                found=True,
                matched_text=best_text,
                start_time=start_time,
                end_time=end_time,
                confidence=round(composite_stt_confidence, 4),
                words=matched_words,
            )

        return STTResult(found=False, confidence=round(best_score, 4))
