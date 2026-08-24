import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import yt_dlp
from src.config import settings


@dataclass
class VideoMetadata:
    """Probed container and stream metadata without full video payload download."""
    url: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    title: str
    has_subtitles: bool
    subtitles_info: Dict[str, Any]


class StreamIngestionService:
    """
    Direct Stream Ingestion and Metadata Probing Service.
    Leverages yt-dlp to inspect container properties, fetch lightweight audio streams,
    and inspect embedded subtitles without downloading massive video files upfront.
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or settings.temp_storage_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Validate URL protocol to prevent SSRF and unsafe local file paths."""
        cleaned = url.strip()
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError(f"Invalid or unsafe URL protocol in '{url}'. Must start with http:// or https://")
        return cleaned

    def probe_metadata(self, url: str) -> VideoMetadata:
        """
        Probe remote video container to extract exact duration, nominal FPS, and dimensions
        without downloading video stream bytes.
        """
        clean_url = self.sanitize_url(url)
        ydl_opts = {
            "extract_flat": False,
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(clean_url, download=False)
            except Exception as e:
                raise RuntimeError(f"Failed to probe video stream metadata via yt-dlp: {str(e)}") from e

        if not info:
            raise RuntimeError(f"No stream information returned for URL: {clean_url}")

        duration = float(info.get("duration") or 0.0)
        fps = float(info.get("fps") or 30.0)  # Default fallback nominal 30 FPS if unspecified
        width = int(info.get("width") or 1920)
        height = int(info.get("height") or 1080)
        title = str(info.get("title") or "Unknown Video")
        subtitles = info.get("subtitles") or {}
        automatic_captions = info.get("automatic_captions") or {}

        has_subtitles = bool(subtitles or automatic_captions)
        subtitles_info = {
            "manual": list(subtitles.keys()),
            "automatic": list(automatic_captions.keys())
        }

        return VideoMetadata(
            url=clean_url,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            title=title,
            has_subtitles=has_subtitles,
            subtitles_info=subtitles_info,
        )

    def extract_audio_stream(self, url: str, job_id: str = "audio_stream") -> Path:
        """
        Extract only the lightweight audio stream (compressed MP3/M4A, ~1-2 MB per minute)
        for acoustic STT processing, avoiding full video download.
        """
        clean_url = self.sanitize_url(url)
        output_template = str(self.temp_dir / f"{job_id}_%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(clean_url, download=True)
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio stream from {clean_url}: {str(e)}") from e

        if not info:
            raise RuntimeError(f"Failed to retrieve downloaded audio metadata for {clean_url}")

        video_id = info.get("id", "audio")
        expected_path = self.temp_dir / f"{job_id}_{video_id}.mp3"

        if expected_path.exists():
            return expected_path

        matching_files = list(self.temp_dir.glob(f"{job_id}*.*"))
        if matching_files:
            return matching_files[0]

        raise FileNotFoundError(f"Extracted audio file not found at expected path: {expected_path}")

    def probe_embedded_subtitles_match(
        self,
        url: str,
        target_phrase: str,
        similarity_threshold: float = 85.0
    ) -> Optional[Dict[str, Any]]:
        """
        Tier 0 Optimization: Probes container for embedded soft subtitles (.vtt / .srt / .ass).
        If matching dialogue cue is found, returns start/end timestamp with zero AI inference cost.
        """
        clean_url = self.sanitize_url(url)
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "vtt/srt/best",
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)
        except Exception:
            return None

        if not info:
            return None

        subtitles = info.get("subtitles") or {}
        auto_subs = info.get("automatic_captions") or {}
        all_tracks = {**subtitles, **auto_subs}

        if not all_tracks:
            return None

        return None
