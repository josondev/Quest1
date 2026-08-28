from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    AMBIGUOUS = "ambiguous"


class TierType(str, Enum):
    TIER_0_SUBTITLE = "Tier 0: Embedded Subtitle Match"
    TIER_1_STT = "Tier 1: STT Acoustic Match"
    TIER_2_SPARSE_OCR = "Tier 2: Sparse Timeline OCR Match"
    TIER_3_DENSE_OCR = "Tier 3: Dense Onset OCR Confirmation"
    TIER_4_VLM_FALLBACK = "Tier 4: VLM Arbiter Fallback"


class VideoMetadata(BaseModel):
    url: str

    duration_seconds: float = Field(
        default=0.0,
        ge=0.0
    )

    fps: float = Field(
        default=25.0,
        gt=0.0
    )

    total_frames: int = Field(
        default=0,
        ge=0
    )

    has_subtitles: bool = False

    is_local: bool = False

    stream_path: Optional[str] = None



class JobRequest(BaseModel):

    url: Optional[str] = None

    local_file_path: Optional[str] = None

    target_text: str = Field(
        ...,
        min_length=1
    )


    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v):

        if v is None:
            return v


        clean_url = v.strip()


        if not (
            clean_url.startswith("http://")
            or clean_url.startswith("https://")
            or Path(clean_url).exists()
        ):

            raise ValueError(
                "URL must be http/https or valid local path"
            )


        return clean_url



    @field_validator("target_text")
    @classmethod
    def validate_target_text(cls,v):

        text = v.strip()

        if not text:
            raise ValueError(
                "Target dialogue cannot be empty"
            )

        return text



class BoundingBox(BaseModel):

    ymin: float = Field(
        default=0,
        ge=0,
        le=1
    )

    xmin: float = Field(
        default=0,
        ge=0,
        le=1
    )

    ymax: float = Field(
        default=1,
        ge=0,
        le=1
    )

    xmax: float = Field(
        default=1,
        ge=0,
        le=1
    )



class WordTimestamp(BaseModel):

    word: str

    start: float

    end: float

    probability: float = Field(
        default=1,
        ge=0,
        le=1
    )



class SubtitleMatchResult(BaseModel):

    start_time: float

    end_time: float

    matched_text: str

    similarity_score: float = Field(
        ge=0,
        le=1
    )

    track_language: str="en"

    is_auto_generated: bool=False



class STTResult(BaseModel):

    found: bool=False

    matched_text: Optional[str]=None

    start_time: Optional[float]=None

    end_time: Optional[float]=None

    confidence: float=Field(
        default=0,
        ge=0,
        le=1
    )

    words: List[WordTimestamp]=Field(
        default_factory=list
    )



class OCRResult(BaseModel):

    frame_index:int

    timestamp_seconds:float

    detected_text:str

    confidence:float=Field(
        default=1,
        ge=0,
        le=1
    )

    bounding_box:Optional[BoundingBox]=None



class CandidateFrame(BaseModel):

    candidate_id: str

    timestamp_seconds: float

    frame_number: int

    image_path: Optional[str] = None

    ocr_detected_text: Optional[str] = None

    ocr_confidence: float = 0.0



class VLMDecision(BaseModel):

    selected_candidate_id:str

    exact_detected_text:str=""


    bounding_box:Optional[BoundingBox]=None


    confidence_score:float=Field(
        default=0,
        ge=0,
        le=1
    )


    reasoning:Optional[str]=""



class JobStatusResponse(BaseModel):

    job_id:str

    status:JobStatus


    target_dialogue:Optional[str]=None


    formatted_timestamp:Optional[str]=None


    timestamp_seconds:Optional[float]=None


    # FINAL FRAME NUMBER
    frame_number:Optional[int]=None


    fps:Optional[float]=None


    confidence_score:Optional[float]=None


    tier_executed:Optional[str]=None


    # FINAL EXTRACTED TEXT
    extracted_text:Optional[str]=None


    # FINAL IMAGE
    frame_image_path:Optional[str]=None


    error_message:Optional[str]=None



JobResponse = JobStatusResponse



class DetectionResult(BaseModel):
    job_id: str

    status: JobStatus

    target_dialogue: str

    # ==========================
    # REQUIRED OUTPUT METADATA
    # ==========================

    timestamp_seconds: Optional[float] = None

    formatted_timestamp: Optional[str] = None

    frame_number: Optional[int] = None

    extracted_text: Optional[str] = None


    # ==========================
    # CONFIDENCE
    # ==========================

    confidence_score: Optional[float] = None


    # ==========================
    # IMAGE OUTPUT
    # ==========================

    frame_image_path: Optional[str] = None


    # ==========================
    # DEBUG / PIPELINE INFO
    # ==========================

    tier_executed: Optional[TierType] = None

    error_message: Optional[str] = None


    model_config = {
        "from_attributes": True,
        "exclude_none": False,
    }