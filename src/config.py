from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings and Multi-Tier Pipeline Configuration.
    Loaded dynamically from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ==========================================
    # OCR Engine Configuration (Tier 2/3)
    # ==========================================
    mistral_api_key: str = Field(
        default="",
        description="Mistral AI API Key for Mistral OCR service"
    )

    # ==========================================
    # Speech-to-Text STT Configuration (Tier 1)
    # ==========================================
    whisper_provider: Literal["groq", "huggingface"] = Field(
        default="groq",
        description="STT Provider (groq or huggingface)"
    )
    whisper_model_name: str = Field(
        default="whisper-large-v3",
        description="Model name for Whisper transcription"
    )
    groq_api_key: str = Field(
        default="",
        description="Groq API Key for fast Whisper inference with word timestamps"
    )
    hf_token: str = Field(
        default="",
        description="Hugging Face User Access Token for Inference API"
    )

    # ==========================================
    # Fallback VLM Arbiter Configuration (Tier 4)
    # ==========================================
    vlm_provider: Literal["nvidia", "huggingface"] = Field(
        default="nvidia",
        description="Vision-Language Model provider"
    )
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA NIM API Key from build.nvidia.com"
    )
    nvidia_vlm_model: str = Field(
        default="meta/llama-3.2-11b-vision-instruct",
        description="NVIDIA NIM VLM model identifier"
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM OpenAI-compatible base URL"
    )

    # ==========================================
    # Confidence Fusion Weights & Thresholds
    # ==========================================
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum composite confidence score to accept without VLM fallback"
    )
    weight_similarity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Weight for fuzzy text similarity"
    )
    weight_ocr_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for OCR optical character recognition confidence"
    )
    weight_temporal_alignment: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for temporal audio-visual overlap alignment"
    )

    # ==========================================
    # Search & Sampling Windows
    # ==========================================
    subtitle_search_padding_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Padding window [t_start - pad, t_end + pad] for subtitle lead/lag"
    )
    sparse_ocr_fps: float = Field(
        default=1.0,
        ge=0.2,
        le=5.0,
        description="Sampling rate (FPS) for Tier 2 sparse timeline scan"
    )

    # ==========================================
    # Storage & Network
    # ==========================================
    artifact_storage_dir: Path = Field(
        default=Path("./artifacts"),
        description="Directory for persisting verified frame images & cropped dialogue"
    )
    temp_storage_dir: Path = Field(
        default=Path("./temp_data"),
        description="Directory for temporary audio slices and frame buffers"
    )
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")

    @property
    def artifacts_dir(self) -> Path:
        """Alias helper for artifact storage directory compatibility."""
        return self.artifact_storage_dir

    @field_validator("artifact_storage_dir", "temp_storage_dir", mode="after")
    @classmethod
    def ensure_directories_exist(cls, path_value: Path) -> Path:
        """Ensure storage directories are created automatically upon startup."""
        path_value.mkdir(parents=True, exist_ok=True)
        return path_value


settings = Settings()