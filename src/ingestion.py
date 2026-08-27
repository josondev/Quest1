import json
import logging
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yt_dlp
from rapidfuzz import fuzz
from src.config import settings
from src.models.schemas import SubtitleMatchResult, VideoMetadata

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Domain exception raised when media ingestion or metadata probing fails."""
    pass


class StreamIngestionService:
    """Service responsible for media probing, subtitle processing, and audio/frame extraction."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or settings.artifacts_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by removing punctuation, lowercasing, and stripping whitespace."""
        if not text:
            return ""
        clean = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", clean).strip().lower()

    @staticmethod
    def _parse_vtt_timestamp(ts_str: str) -> float:
        """Convert standard WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to total seconds float."""
        ts_str = ts_str.strip().replace(",", ".")
        parts = ts_str.split(":")
        try:
            if len(parts) == 3:
                hrs, mins, secs = float(parts[0]), float(parts[1]), float(parts[2])
                return hrs * 3600.0 + mins * 60.0 + secs
            elif len(parts) == 2:
                mins, secs = float(parts[0]), float(parts[1])
                return mins * 60.0 + secs
            return float(ts_str)
        except Exception:
            return 0.0

    def _parse_vtt_file(self, vtt_path: Path) -> List[Dict[str, Any]]:
        """Parse WebVTT file content into a list of structured cues."""
        if not vtt_path.exists():
            return []
        
        cues = []
        content = vtt_path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n\s*\n", content)

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            time_line_idx = -1
            for idx, line in enumerate(lines):
                if "-->" in line:
                    time_line_idx = idx
                    break

            if time_line_idx != -1:
                time_line = lines[time_line_idx]
                start_str, end_str = [t.strip() for t in time_line.split("-->")[:2]]
                end_str = end_str.split()[0]

                start_sec = self._parse_vtt_timestamp(start_str)
                end_sec = self._parse_vtt_timestamp(end_str)

                text_lines = lines[time_line_idx + 1 :]
                raw_text = " ".join(text_lines)
                clean_text = re.sub(r"<[^>]+>", "", raw_text)

                if clean_text:
                    cues.append({
                        "start_seconds": start_sec,
                        "end_seconds": end_sec,
                        "text": clean_text
                    })

        return cues

    def _match_phrase_in_cues(
        self,
        target_phrase: str,
        cues: List[Dict[str, Any]],
        similarity_threshold: float = 85.0,
        max_gap_seconds: float = 2.0,
        track_language: str = "en",
        is_auto_generated: bool = False,
    ) -> Optional[SubtitleMatchResult]:
        """Align a target phrase across cue sequences in WebVTT subtitle tracks."""
        if not target_phrase or not cues:
            return None

        norm_target = self._normalize_text(target_phrase)
        if not norm_target:
            return None

        target_words = norm_target.split()

        # Only allow single cue matches if the complete phrase exists
        for cue in cues:
            norm_cue = self._normalize_text(cue["text"])
            if not norm_cue:
                continue

            if norm_target == norm_cue or norm_target in norm_cue:
                return SubtitleMatchResult(
                    start_time=cue["start_seconds"],
                    end_time=cue["end_seconds"],
                    matched_text=cue["text"],
                    similarity_score=1.0,
                    track_language=track_language,
                    is_auto_generated=is_auto_generated,
                )

        # Multi-cue sliding window alignment
        for i in range(len(cues)):
            accumulated_text = ""
            start_time = cues[i]["start_seconds"]
            last_end_time = cues[i]["end_seconds"]

            for j in range(i, len(cues)):
                curr_cue = cues[j]
                
                if j > i and (curr_cue["start_seconds"] - last_end_time) > max_gap_seconds:
                    break

                accumulated_text = (accumulated_text + " " + curr_cue["text"]).strip()
                last_end_time = curr_cue["end_seconds"]
                norm_acc = self._normalize_text(accumulated_text)

                if norm_target in norm_acc:
                    return SubtitleMatchResult(
                        start_time=start_time,
                        end_time=last_end_time,
                        matched_text=accumulated_text,
                        similarity_score=1.0,
                        track_language=track_language,
                        is_auto_generated=is_auto_generated,
                    )

                score = fuzz.ratio(norm_target, norm_acc)
                if score >= similarity_threshold:
                    return SubtitleMatchResult(
                        start_time=start_time,
                        end_time=last_end_time,
                        matched_text=accumulated_text,
                        similarity_score=round(score / 100.0, 4),
                        track_language=track_language,
                        is_auto_generated=is_auto_generated,
                    )

                if len(norm_acc.split()) >= len(target_words) + 5:
                    break

        return None

    def probe_embedded_subtitles_match(
        self, url: str, target_phrase: str, target_dir: Optional[Path] = None
    ) -> Optional[SubtitleMatchResult]:
        """Download subtitle track and evaluate phrase match."""
        target_dir = target_dir or self.cache_dir
        sub_file, is_auto, lang = self._download_subtitle_file(url, target_dir)
        if not sub_file or not sub_file.exists():
            return None

        cues = self._parse_vtt_file(sub_file)
        return self._match_phrase_in_cues(
            target_phrase=target_phrase,
            cues=cues,
            track_language=lang,
            is_auto_generated=is_auto,
        )

    def probe_metadata(self, url_or_path: str) -> VideoMetadata:
        """Extract metadata for remote URLs or local files."""
        local_path = Path(url_or_path)
        if local_path.exists() and local_path.is_file():
            return self._probe_local_file(local_path)
        return self._probe_remote_url(url_or_path)

    def _probe_local_file(self, file_path: Path) -> VideoMetadata:
        if not file_path.exists():
            raise IngestionError(f"Local file does not exist: {file_path}")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=r_frame_rate,codec_type",
            "-of", "json", str(file_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0.0))
            fps = 25.0
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    rate_str = stream.get("r_frame_rate", "25/1")
                    if "/" in rate_str:
                        num, den = rate_str.split("/")
                        fps = float(num) / float(den) if float(den) > 0 else 25.0
                    break
            return VideoMetadata(
                url=str(file_path),
                duration_seconds=duration,
                fps=fps,
                has_subtitles=False,
                is_local=True,
                stream_path=str(file_path),
            )
        except Exception as exc:
            cap = cv2.VideoCapture(str(file_path))
            try:
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                    duration = frame_count / fps if fps > 0 else 0.0

                    return VideoMetadata(
                        url=str(file_path),
                        duration_seconds=duration,
                        fps=fps,
                        has_subtitles=False,
                        is_local=True,
                        stream_path=str(file_path),
                    )
            finally:
                cap.release()
            raise IngestionError(f"Unable to open local video file {file_path}: {exc}") from exc

    def _probe_remote_url(self, url: str) -> VideoMetadata:
        ydl_opts = {"quiet": True, "no_warnings": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise IngestionError(f"Probing metadata failed for URL: {url}")
                
                duration = float(info.get("duration", 0.0) or 0.0)
                fps = float(info.get("fps", 25.0) or 25.0)
                subs = info.get("subtitles") or info.get("automatic_captions") or {}
                
                stream_url = None
                formats = info.get("formats", [])
                for fmt in formats:
                    if fmt.get("vcodec") != "none" and fmt.get("url"):
                        stream_url = fmt.get("url")
                        break
                
                if not stream_url:
                    stream_url = info.get("url", url)

                return VideoMetadata(
                    url=url,
                    duration_seconds=duration,
                    fps=fps,
                    has_subtitles=bool(subs),
                    is_local=False,
                    stream_path=stream_url,
                )
        except Exception as exc:
            logger.error("Failed probing remote URL %s: %s", url, exc)
            raise IngestionError(f"Probing metadata failed for remote URL {url}: {exc}") from exc

    def _download_subtitle_file(
        self, url: str, target_dir: Path
    ) -> Tuple[Optional[Path], bool, str]:
        """Download subtitle track using yt_dlp or direct URL fallback."""
        ydl_opts = {"quiet": True, "no_warnings": True, "writesubtitles": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False) or {}
                subs = info.get("subtitles") or {}
                auto_subs = info.get("automatic_captions") or {}

                lang_key = "en"
                is_auto = False
                vtt_url = None

                if "en" in subs and subs["en"]:
                    vtt_url = subs["en"][0].get("url")
                elif "en" in auto_subs and auto_subs["en"]:
                    vtt_url = auto_subs["en"][0].get("url")
                    is_auto = True

                if vtt_url:
                    vtt_path = target_dir / f"sub_{lang_key}.vtt"
                    try:
                        req = urllib.request.Request(
                            vtt_url,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                "Referer": "https://ok.ru/",
                            },
                        )
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            data = resp.read()
                            if data:
                                with open(vtt_path, "wb") as out:
                                    out.write(data)
                                if vtt_path.exists() and vtt_path.stat().st_size > 0:
                                    return (vtt_path, is_auto, lang_key)
                    except Exception as dl_exc:
                        logger.warning(
                            "Direct subtitle URL download failed, falling back to yt-dlp: %s",
                            dl_exc,
                        )

                ydl_opts_dl = {
                    "quiet": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitlesformat": "vtt",
                    "outtmpl": str(target_dir / "sub_test"),
                }
                with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl_dl:
                    ydl_dl.download([url])

                vtt_files = list(target_dir.glob("*.vtt"))
                if vtt_files:
                    return (vtt_files[0], is_auto, lang_key)
        except Exception as exc:
            logger.warning("Failed fetching soft subtitle text: %s", exc)

        return (None, False, "")

    def extract_audio_stream(
        self,
        stream_url_or_path: str,
        output_wav: Optional[Path] = None,
        job_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Extract audio to 16kHz mono PCM WAV format."""
        source_path = Path(stream_url_or_path)
        if source_path.exists() and source_path.is_file():
            return source_path

        if output_wav is None:
            out_dir = Path(output_dir or settings.artifacts_dir / (job_id or "default_job"))
            out_dir.mkdir(parents=True, exist_ok=True)
            output_wav = out_dir / f"audio_{job_id or 'default_job'}_audio.wav"

        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        if stream_url_or_path.startswith(("http://", "https://")):
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }],
                "outtmpl": str(output_wav.with_suffix("")),
                "quiet": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([stream_url_or_path])
                if output_wav.exists():
                    return output_wav
            except Exception as exc:
                logger.warning("yt-dlp audio download failed, falling back to FFmpeg: %s", exc)

        cmd = ["ffmpeg", "-y"]
        if stream_url_or_path.startswith(("http://", "https://")):
            cmd.extend([
                "-headers", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\nReferer: https://ok.ru/\r\n"
            ])

        cmd.extend([
            "-i", stream_url_or_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_wav)
        ])

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_wav
        except Exception as exc:
            logger.error("Audio extraction failed: %s", exc)
            raise IngestionError(f"Audio extraction failed: {exc}") from exc

    def extract_frame_on_demand(
        self, stream_url: str, timestamp_seconds: float, output_jpg: Path
    ) -> bool:
        """Extract a single visual frame at a target timestamp."""
        output_jpg = Path(output_jpg)
        output_jpg.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-y", "-ss", str(timestamp_seconds)]
        if stream_url.startswith(("http://", "https://")):
            cmd.extend([
                "-headers", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\nReferer: https://ok.ru/\r\n"
            ])

        cmd.extend([
            "-i", stream_url,
            "-vframes", "1", "-q:v", "2",
            str(output_jpg)
        ])

        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0 and output_jpg.exists()
        except Exception as exc:
            logger.warning("Frame extraction failed: %s", exc)
            return False