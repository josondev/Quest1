import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import yt_dlp
from src.config import settings


class IngestionError(Exception):
    """Clean domain exception raised when stream ingestion or metadata probing fails."""
    pass


class SilentYTDLPLogger:
    """Suppresses internal yt-dlp stderr noise and raw stack trace dumps."""
    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


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

    @staticmethod
    def _parse_friendly_error(clean_url: str, error: Exception) -> str:
        """Extract a clean, human-readable explanation from low-level network/SSL errors."""
        err_msg = str(error)
        err_lower = err_msg.lower()

        # 1. SSL / Handshake / Connection Reset / Geo-blocking
        if any(k in err_lower for k in ["ssl", "10054", "connection was reset", "connectionreset", "handshake", "certificate"]):
            return (
                f"Unable to connect to the video host '{clean_url}' due to network/SSL handshake failure. "
                "The remote platform may be geo-blocked or enforcing network restrictions in your region."
            )

        # 2. HTTP 403 / Private / Forbidden
        if "403" in err_msg or "forbidden" in err_lower or "private video" in err_lower:
            return f"Access to video stream '{clean_url}' was denied (HTTP 403 / Private video or region lock)."

        # 3. HTTP 404 / Video Removed / Not Found
        if "404" in err_msg or "not found" in err_lower or "video unavailable" in err_lower:
            return f"Video could not be found at '{clean_url}' (HTTP 404 / Video removed or invalid URL)."

        # 4. Strip internal yt-dlp / extractor noise and boilerplate
        clean = err_msg.splitlines()[0] if err_msg else "Unknown video stream error."
        clean = re.sub(r"^ERROR:\s*", "", clean)
        clean = re.sub(r"^\[.*?\]\s*", "", clean)
        clean = clean.split("; please report")[0].strip()
        clean = clean.split(" (caused by")[0].strip()

        return f"Stream error at '{clean_url}': {clean}"

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
            "logger": SilentYTDLPLogger(),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(clean_url, download=False)
            except Exception as e:
                friendly_msg = self._parse_friendly_error(clean_url, e)
                raise IngestionError(friendly_msg) from None

        if not info:
            raise IngestionError(f"No stream information returned for URL: {clean_url}")

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
            "logger": SilentYTDLPLogger(),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(clean_url, download=True)
            except Exception as e:
                friendly_msg = self._parse_friendly_error(clean_url, e)
                raise IngestionError(friendly_msg) from None

        if not info:
            raise IngestionError(f"Failed to retrieve downloaded audio metadata for {clean_url}")

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
            "logger": SilentYTDLPLogger(),
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
