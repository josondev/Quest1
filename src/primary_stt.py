import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rapidfuzz import fuzz

from src.config import settings
from src.models.schemas import STTResult, WordTimestamp

# Keep these available at module level because the test suite patches them.
try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore


logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for robust acoustic dialogue matching.

    - lowercase
    - punctuation -> spaces
    - collapse repeated whitespace
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SpeechToTextService:
    """
    Provider-agnostic STT service.

    Supported providers:
        - groq
        - huggingface

    The clients are initialized lazily so importing the API does not
    immediately contact external services or load a local Whisper model.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        # IMPORTANT:
        # Use the configured settings INSTANCE, not the Settings class.
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

        logger.info(
            "SpeechToTextService configured: provider=%s model=%s",
            self.provider,
            self.model_name,
        )

    # ==========================================================
    # LAZY CLIENT INITIALIZATION
    # ==========================================================

    def _get_groq_client(self) -> Any:
        """
        Lazily initialize Groq client.
        """
        if self._groq_client is not None:
            return self._groq_client

        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Set GROQ_API_KEY in .env."
            )

        if Groq is None:
            raise ImportError(
                "groq package is required for Groq STT. "
                "Install it with: pip install groq"
            )

        self._groq_client = Groq(
            api_key=self.groq_api_key
        )

        logger.info("Groq STT client initialized.")

        return self._groq_client

    def _get_hf_client(self) -> Any:
        """
        Lazily initialize Hugging Face client.
        """
        if self._hf_client is not None:
            return self._hf_client

        if InferenceClient is None:
            raise ImportError(
                "huggingface_hub is required for Hugging Face STT. "
                "Install it with: pip install huggingface_hub"
            )

        self._hf_client = InferenceClient(
            token=self.hf_token if self.hf_token else None
        )

        logger.info("Hugging Face STT client initialized.")

        return self._hf_client

    # ==========================================================
    # AUDIO VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_audio_file(audio_path: Path) -> None:
        """
        Validate that the input audio exists and is non-empty.
        """
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist: {audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Audio path is not a file: {audio_path}"
            )

        if audio_path.stat().st_size <= 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

    # ==========================================================
    # PUBLIC TRANSCRIPTION API
    # ==========================================================

    def transcribe_audio(
        self,
        audio_path: Path,
    ) -> List[WordTimestamp]:
        """
        Transcribe an audio file and return word-level timestamps.
        """
        audio_path = Path(audio_path)

        self._validate_audio_file(audio_path)

        logger.info(
            "Starting STT: provider=%s file=%s size=%d bytes",
            self.provider,
            audio_path,
            audio_path.stat().st_size,
        )

        if self.provider == "groq":
            return self._transcribe_with_groq(audio_path)

        if self.provider == "huggingface":
            return self._transcribe_with_hf(audio_path)

        raise ValueError(
            f"Unsupported provider: {self.provider}"
        )

    # ==========================================================
    # GROQ
    # ==========================================================

    def _transcribe_with_groq(
        self,
        audio_path: Path,
    ) -> List[WordTimestamp]:
        """
        Transcribe using Groq Whisper with word timestamps.
        """
        client = self._get_groq_client()

        logger.info(
            "Sending audio to Groq: model=%s",
            self.model_name,
        )

        try:
            with audio_path.open("rb") as audio_file:
                transcription = (
                    client.audio.transcriptions.create(
                        file=(
                            audio_path.name,
                            audio_file.read(),
                        ),
                        model=self.model_name,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        temperature=0.0,
                    )
                )
        except Exception as exc:
            logger.exception(
                "Groq transcription failed."
            )
            raise RuntimeError(
                f"Groq STT failed: {exc}"
            ) from exc

        words: List[WordTimestamp] = []

        raw_words = (
            getattr(transcription, "words", None)
            or []
        )

        for raw_word in raw_words:
            try:
                if isinstance(raw_word, dict):
                    word_text = str(
                        raw_word.get("word", "")
                    ).strip()

                    start = float(
                        raw_word.get("start", 0.0)
                        or 0.0
                    )

                    end = float(
                        raw_word.get("end", start)
                        or start
                    )

                    probability_raw = raw_word.get(
                        "probability",
                        raw_word.get(
                            "avg_logprob",
                            1.0,
                        ),
                    )

                else:
                    word_text = str(
                        getattr(
                            raw_word,
                            "word",
                            "",
                        )
                    ).strip()

                    start = float(
                        getattr(
                            raw_word,
                            "start",
                            0.0,
                        )
                        or 0.0
                    )

                    end = float(
                        getattr(
                            raw_word,
                            "end",
                            start,
                        )
                        or start
                    )

                    probability_raw = getattr(
                        raw_word,
                        "probability",
                        1.0,
                    )

                try:
                    probability = float(
                        probability_raw
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    probability = 1.0

                probability = max(
                    0.0,
                    min(1.0, probability),
                )

                if word_text:
                    words.append(
                        WordTimestamp(
                            word=word_text,
                            start=start,
                            end=end,
                            probability=probability,
                        )
                    )

            except Exception as exc:
                logger.warning(
                    "Skipping malformed Groq word: %s",
                    exc,
                )

        logger.info(
            "Groq transcription complete: %d words",
            len(words),
        )

        return words

    # ==========================================================
    # HUGGING FACE
    # ==========================================================

    def _transcribe_with_hf(
        self,
        audio_path: Path,
    ) -> List[WordTimestamp]:
        """
        Transcribe using Hugging Face inference.
        """
        client = self._get_hf_client()

        logger.info(
            "Sending audio to Hugging Face: model=%s",
            self.model_name,
        )

        try:
            with audio_path.open("rb") as audio_file:
                result = client.automatic_speech_recognition(
                    audio=audio_file.read(),
                    model=self.model_name,
                )
        except Exception as exc:
            logger.exception(
                "Hugging Face transcription failed."
            )
            raise RuntimeError(
                f"Hugging Face STT failed: {exc}"
            ) from exc

        if isinstance(result, dict):
            chunks = result.get(
                "chunks",
                [],
            )
        else:
            chunks = (
                getattr(
                    result,
                    "chunks",
                    None,
                )
                or []
            )

        words: List[WordTimestamp] = []

        for chunk in chunks:
            try:
                if isinstance(chunk, dict):
                    text = str(
                        chunk.get(
                            "text",
                            "",
                        )
                    ).strip()

                    timestamp = chunk.get(
                        "timestamp",
                        (0.0, 0.0),
                    )

                else:
                    text = str(
                        getattr(
                            chunk,
                            "text",
                            "",
                        )
                    ).strip()

                    timestamp = getattr(
                        chunk,
                        "timestamp",
                        (0.0, 0.0),
                    )

                if not text:
                    continue

                start = 0.0
                end = 0.0

                if (
                    isinstance(timestamp, (tuple, list))
                    and len(timestamp) >= 2
                ):
                    try:
                        start = float(
                            timestamp[0]
                            if timestamp[0] is not None
                            else 0.0
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        start = 0.0

                    try:
                        end = float(
                            timestamp[1]
                            if timestamp[1] is not None
                            else start
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        end = start

                words.append(
                    WordTimestamp(
                        word=text,
                        start=start,
                        end=end,
                        probability=0.9,
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Skipping malformed HF chunk: %s",
                    exc,
                )

        logger.info(
            "Hugging Face transcription complete: %d chunks",
            len(words),
        )

        return words

    # ==========================================================
    # TARGET DIALOGUE ALIGNMENT
    # ==========================================================

    def align_target_dialogue(
        self,
        words: List[WordTimestamp],
        target_dialogue: str,
        min_similarity_threshold: float = 0.70,
    ) -> STTResult:
        """
        Find the best order-preserving matching interval.

        The comparison uses fuzz.ratio() rather than token-set
        matching because dialogue order matters.
        """
        if not words:
            return STTResult(
                found=False
            )

        if not target_dialogue:
            return STTResult(
                found=False
            )

        if not target_dialogue.strip():
            return STTResult(
                found=False
            )

        normalized_target = normalize_text(
            target_dialogue
        )

        if not normalized_target:
            return STTResult(
                found=False
            )

        target_tokens = normalized_target.split()

        target_count = len(
            target_tokens
        )

        total_words = len(words)

        if total_words < target_count:
            return STTResult(
                found=False
            )

        logger.info(
            "Aligning STT: transcript_words=%d target_tokens=%d",
            total_words,
            target_count,
        )

        best_score = 0.0
        best_window: Optional[
            Tuple[int, int]
        ] = None

        # Allow a modest amount of ASR insertion/deletion.
        min_window = max(
            1,
            target_count - 2,
        )

        max_window = min(
            total_words,
            target_count + 3,
        )

        for window_size in range(
            min_window,
            max_window + 1,
        ):
            for start_idx in range(
                0,
                total_words - window_size + 1,
            ):
                end_idx = (
                    start_idx
                    + window_size
                )

                window_words = words[
                    start_idx:end_idx
                ]

                candidate_text = " ".join(
                    word.word
                    for word in window_words
                )

                candidate_norm = normalize_text(
                    candidate_text
                )

                if not candidate_norm:
                    continue

                # Order-aware dialogue similarity.
                score = (
                    fuzz.ratio(
                        normalized_target,
                        candidate_norm,
                    )
                    / 100.0
                )

                if score > best_score:
                    best_score = score
                    best_window = (
                        start_idx,
                        end_idx,
                    )

        if (
            best_window is None
            or best_score < min_similarity_threshold
        ):
            logger.info(
                "No STT match found. Best similarity=%.4f",
                best_score,
            )

            return STTResult(
                found=False,
                confidence=round(
                    best_score,
                    4,
                ),
            )

        start_idx, end_idx = best_window

        matched_words = words[
            start_idx:end_idx
        ]

        if not matched_words:
            return STTResult(
                found=False,
                confidence=round(
                    best_score,
                    4,
                ),
            )

        start_time = float(
            matched_words[0].start
        )

        end_time = float(
            matched_words[-1].end
        )

        probabilities = []

        for word in matched_words:
            try:
                probability = float(
                    word.probability
                )
            except (
                TypeError,
                ValueError,
            ):
                probability = 1.0

            probabilities.append(
                max(
                    0.0,
                    min(
                        1.0,
                        probability,
                    ),
                )
            )

        mean_probability = (
            sum(probabilities)
            / len(probabilities)
            if probabilities
            else 1.0
        )

        composite_confidence = (
            best_score * 0.70
            + mean_probability * 0.30
        )

        matched_text = " ".join(
            word.word
            for word in matched_words
        )

        result = STTResult(
            found=True,
            matched_text=matched_text,
            start_time=start_time,
            end_time=end_time,
            confidence=round(
                composite_confidence,
                4,
            ),
            words=matched_words,
        )

        logger.info(
            "STT match found: start=%.3fs end=%.3fs "
            "similarity=%.4f confidence=%.4f text='%s'",
            start_time,
            end_time,
            best_score,
            result.confidence,
            matched_text,
        )

        return result