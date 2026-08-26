from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


class TierType(str, Enum):
    TIER_0_SUBTITLE = "Tier 0: Embedded Subtitle Match"
    TIER_1_STT = "Tier 1: STT Acoustic Match"
    TIER_2_SPARSE_OCR = "Tier 2: Sparse Timeline OCR Match"
    TIER_3_DENSE_OCR = "Tier 3: Dense Onset OCR Confirmation"
    TIER_4_VLM_FALLBACK = "Tier 4: VLM Arbiter Fallback"


class VideoMetadata(BaseModel):
    """Container for video container metadata extracted during ingestion probing."""
    url: str = Field(..., description="Target media URL or local video path")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Total duration in seconds")
    fps: float = Field(default=25.0, gt=0.0, description="Frames per second")
    has_subtitles: bool = Field(default=False, description="True if embedded subtitle tracks are available")
    is_local: bool = Field(default=False, description="True if local file, False if remote network URL")
    stream_path: Optional[str] = Field(default=None, description="Direct playable stream URL or local path for VideoCapture")


class JobRequest(BaseModel):
    """Payload received from user/client to locate target dialogue."""
    url: str = Field(..., description="Target media URL (e.g. YouTube, direct MP4/stream URL)")
    target_text: str = Field(..., min_length=1, description="Target dialogue phrase to locate dynamically within the video")

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        clean_url = v.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://") or Path(clean_url).exists()):
            raise ValueError("URL must begin with http:// or https:// or be an existing local path")
        return clean_url

    @field_validator("target_text")
    @classmethod
    def validate_target_text(cls, v: str) -> str:
        clean_text = v.strip()
        if not clean_text:
            raise ValueError("Target dialogue text cannot be empty or whitespace only")
        return clean_text


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates [0.0 to 1.0] for subtitle ROI."""
    ymin: float = Field(default=0.0, ge=0.0, le=1.0)
    xmin: float = Field(default=0.0, ge=0.0, le=1.0)
    ymax: float = Field(default=1.0, ge=0.0, le=1.0)
    xmax: float = Field(default=1.0, ge=0.0, le=1.0)


class WordTimestamp(BaseModel):
    """Word-level acoustic timestamp from Whisper STT."""
    word: str
    start: float
    end: float
    probability: float = Field(default=1.0, ge=0.0, le=1.0)


class SubtitleMatchResult(BaseModel):
    """Tier 0 result: matched embedded subtitle cue with confidence score."""
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")
    matched_text: str = Field(..., description="Raw text extracted from subtitle cue")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Fuzzy match confidence (0.0-1.0)")
    track_language: str = Field(default="en", description="Language code (e.g. en-US, en-GB)")
    is_auto_generated: bool = Field(default=False, description="True if auto-captions, False if manual")


class STTResult(BaseModel):
    """Result returned by Tier 1 STT search."""
    found: bool
    matched_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    words: List[WordTimestamp] = Field(default_factory=list)


class OCRResult(BaseModel):
    """Result from Mistral OCR on a specific frame."""
    frame_index: int
    timestamp_seconds: float
    detected_text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None


class CandidateFrame(BaseModel):
    """Frame representation passed to Tier 4 VLM Arbiter."""
    candidate_id: str = Field(..., description="Unique identifier (e.g. C1, C2)")
    timestamp_seconds: float
    frame_number: int
    image_path: str
    ocr_detected_text: Optional[str] = None
    ocr_confidence: Optional[float] = None


class VLMDecision(BaseModel):
    """Structured decision returned by Tier 4 VLM Arbiter."""
    selected_candidate_id: str = Field(..., description="ID of selected frame or 'NONE'")
    exact_detected_text: str = Field(default="", description="Verbatim dialogue visible in frame")
    bounding_box: Optional[BoundingBox] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class DetectionResult(BaseModel):
    """Final output payload containing verified frame, timestamp, and visual artifacts."""
    job_id: str
    status: JobStatus
    target_dialogue: str
    timestamp_seconds: Optional[float] = None
    formatted_timestamp: Optional[str] = None
    frame_number: Optional[int] = None
    extracted_text: Optional[str] = None
    confidence_score: Optional[float] = None
    tier_executed: Optional[TierType] = None
    frame_image_path: Optional[str] = None
    cropped_roi_path: Optional[str] = None
    error_message: Optional[str] = None