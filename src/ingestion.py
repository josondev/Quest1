import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import yt_dlp
from rapidfuzz import fuzz

from src.models.schemas import SubtitleMatchResult, VideoMetadata

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Domain exception raised when stream ingestion or metadata probing fails."""
    pass


class StreamIngestionService:
    """
    Stream Ingestion & Metadata Probing Service using yt-dlp and OpenCV.
    Handles network URL probing, WebVTT subtitle downloading/parsing,
    sliding multi-cue matching, local OpenCV video probing, and audio stream extraction.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "extract_flat": False,
        }
        if options:
            self.ydl_opts.update(options)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize case, remove punctuation, and collapse repeated whitespace."""
        text = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _parse_vtt_timestamp(ts_str: str) -> float:
        """Convert VTT timestamp HH:MM:SS.mmm or MM:SS.mmm to decimal seconds."""
        ts_str = ts_str.strip().replace(",", ".")
        parts = ts_str.split(":")
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return h * 3600.0 + m * 60.0 + s
        elif len(parts) == 2:
            m, s = float(parts[0]), float(parts[1])
            return m * 60.0 + s
        return 0.0

    def _parse_vtt_text(self, vtt_text: str) -> List[Dict[str, Any]]:
        """Parse WebVTT string content into timestamped cue dicts."""
        cues: List[Dict[str, Any]] = []
        blocks = re.split(r"\n\s*\n", vtt_text)

        timestamp_re = re.compile(
            r"((?:\d+:)?\d+:\d+(?:[\.,]\d+)?)\s*-->\s*((?:\d+:)?\d+:\d+(?:[\.,]\d+)?)"
        )

        for block in blocks:
            block = block.strip()
            if not block or block.startswith("WEBVTT") or block.startswith("NOTE"):
                continue

            lines = block.splitlines()
            ts_match = None
            text_lines = []

            for line in lines:
                m = timestamp_re.search(line)
                if m:
                    ts_match = m
                elif ts_match:
                    clean_line = re.sub(r"<[^>]+>", "", line).strip()
                    if clean_line:
                        text_lines.append(clean_line)

            if ts_match and text_lines:
                start_sec = self._parse_vtt_timestamp(ts_match.group(1))
                end_sec = self._parse_vtt_timestamp(ts_match.group(2))
                cue_text = " ".join(text_lines)
                cues.append({
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "text": cue_text,
                })

        return cues

    def _parse_vtt_file(self, vtt_path: Path) -> List[Dict[str, Any]]:
        """Parse a WebVTT file from a local file path."""
        if not vtt_path or not Path(vtt_path).exists():
            return []
        try:
            content = Path(vtt_path).read_text(encoding="utf-8", errors="ignore")
            return self._parse_vtt_text(content)
        except Exception as exc:
            logger.warning("Failed reading VTT file %s: %s", vtt_path, exc)
            return []

    def _download_subtitle_file(
        self, url: str, target_dir: Optional[Path] = None
    ) -> Union[Tuple[Optional[Path], bool, str], Optional[Path]]:
        """
        Download English subtitle tracks using yt-dlp to a temporary VTT file.
        Prioritizes manual English tracks over auto-generated captions.
        When target_dir is supplied, the caller owns its lifecycle.
        When target_dir is omitted, this method creates an isolated temporary directory;
        callers must not assume the returned path is persistent.
        """
        clean_url = url.strip()
        if Path(clean_url).exists():
            return None, False, "en"

        work_dir = target_dir or Path(tempfile.mkdtemp(prefix="quest1_vtt_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            ydl_opts_probe = dict(self.ydl_opts)
            ydl_opts_probe["skip_download"] = True
            with yt_dlp.YoutubeDL(ydl_opts_probe) as ydl:
                info = ydl.extract_info(clean_url, download=False)

            if not info:
                return None, False, "en"

            subtitles = info.get("subtitles", {}) or {}
            automatic_captions = info.get("automatic_captions", {}) or {}

            has_manual_en = any(k.startswith("en") for k in subtitles.keys())
            has_auto_en = any(k.startswith("en") for k in automatic_captions.keys())

            if not has_manual_en and not has_auto_en:
                return None, False, "en"

            is_auto = not has_manual_en

            ydl_opts_down = dict(self.ydl_opts)
            ydl_opts_down.update({
                "writesubtitles": has_manual_en,
                "writeautomaticsub": not has_manual_en and has_auto_en,
                "subtitleslangs": ["en.*", "en"],
                "skip_download": True,
                "outtmpl": str(work_dir / "sub_%(id)s"),
            })

            with yt_dlp.YoutubeDL(ydl_opts_down) as ydl:
                ydl.download([clean_url])

            vtt_files = sorted(list(work_dir.glob("*.vtt")))
            if not vtt_files:
                return None, False, "en"

            return vtt_files[0], is_auto, "en"

        except Exception as exc:
            logger.warning("Subtitle download failed for %s: %s", clean_url, exc)
            return None, False, "en"

    def probe_metadata(self, url: str) -> VideoMetadata:
        """Probe media container metadata and stream capabilities via OpenCV or yt-dlp."""
        clean_url = url.strip()
        if Path(clean_url).exists():
            cap = cv2.VideoCapture(clean_url)
            if not cap.isOpened():
                cap.release()
                raise IngestionError(f"Unable to open local video file: {clean_url}")

            try:
                fps_val = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = float(fps_val) if fps_val > 0 else 25.0
                duration = float(frame_count / fps) if frame_count > 0 and fps > 0 else 0.0
            except Exception as cv_err:
                logger.error("OpenCV local probing failed for %s: %s", clean_url, cv_err)
                raise IngestionError(f"Failed to probe local video file '{clean_url}': {cv_err}") from cv_err
            finally:
                cap.release()

            return VideoMetadata(
                url=clean_url,
                duration_seconds=round(duration, 3),
                fps=round(fps, 3),
                has_subtitles=False,
                is_local=True,
                stream_path=clean_url,
            )

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)

            if not info:
                raise IngestionError(f"Could not extract info from URL: {clean_url}")

            duration = float(info.get("duration", 0.0) or 0.0)
            fps = float(info.get("fps", 25.0) or 25.0)
            subtitles = info.get("subtitles", {}) or {}
            automatic_captions = info.get("automatic_captions", {}) or {}
            has_subs = bool(subtitles or automatic_captions)

            formats = info.get("formats", [])
            stream_url = None
            for fmt in formats:
                if fmt.get("vcodec") != "none" and fmt.get("acodec") != "none" and fmt.get("url"):
                    stream_url = fmt["url"]
                    break
            if not stream_url and formats:
                stream_url = formats[0].get("url")

            return VideoMetadata(
                url=clean_url,
                duration_seconds=duration,
                fps=fps,
                has_subtitles=has_subs,
                is_local=False,
                stream_path=stream_url or clean_url,
            )
        except IngestionError:
            raise
        except Exception as exc:
            logger.error("Failed probing metadata for %s: %s", clean_url, exc)
            raise IngestionError(f"Probing metadata failed for URL '{clean_url}': {exc}") from exc

    def probe_embedded_subtitles_match(
        self,
        url: str,
        target_phrase: str,
        similarity_threshold: float = 85.0,
        max_gap_seconds: float = 2.0,
    ) -> Optional[SubtitleMatchResult]:
        """
        Probe embedded or auto-generated subtitle tracks and perform
        sliding multi-cue matching for the target phrase.
        Ensures temporary working directories are cleaned up immediately after parsing.
        """
        with tempfile.TemporaryDirectory(prefix="quest1_vtt_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            sub_info = self._download_subtitle_file(url, target_dir=temp_dir)
            if not sub_info:
                return None

            if isinstance(sub_info, tuple):
                vtt_path, is_auto, track_lang = sub_info
            else:
                vtt_path, is_auto, track_lang = sub_info, False, "en"

            if not vtt_path or not Path(vtt_path).exists():
                return None

            cues = self._parse_vtt_file(vtt_path)
            if not cues:
                return None

            return self._match_phrase_in_cues(
                target_phrase=target_phrase,
                cues=cues,
                similarity_threshold=similarity_threshold,
                max_gap_seconds=max_gap_seconds,
                track_language=track_lang,
                is_auto_generated=is_auto,
            )

    def _match_phrase_in_cues(
        self,
        target_phrase: str,
        cues: List[Dict[str, Any]],
        similarity_threshold: float = 85.0,
        max_gap_seconds: float = 2.0,
        track_language: str = "en",
        is_auto_generated: bool = False,
    ) -> Optional[SubtitleMatchResult]:
        """
        Generalized sliding multi-cue matcher over continuous subtitle cues.
        Ensures target token span matching across cue boundaries while respecting temporal gap limits.
        """
        clean_target = self._normalize_text(target_phrase)
        if not clean_target or not cues:
            return None

        target_words = clean_target.split()
        target_word_count = len(target_words)

        best_result: Optional[SubtitleMatchResult] = None
        best_score: float = 0.0

        for start_idx in range(len(cues)):
            window_parts: List[str] = []
            window_cues: List[Dict[str, Any]] = []

            for current_idx in range(start_idx, len(cues)):
                curr_cue = cues[current_idx]

                if window_cues:
                    prev_cue = window_cues[-1]
                    gap = curr_cue.get("start_seconds", 0.0) - prev_cue.get("end_seconds", 0.0)
                    if gap > max_gap_seconds:
                        break

                window_cues.append(curr_cue)
                window_parts.append(curr_cue.get("text", ""))

                window_raw = " ".join(window_parts)
                normalized_window = self._normalize_text(window_raw)
                window_words = normalized_window.split()

                if len(window_words) < target_word_count:
                    continue

                for span_start in range(len(window_words) - target_word_count + 1):
                    span_words = window_words[span_start : span_start + target_word_count]
                    candidate_span = " ".join(span_words)

                    score = fuzz.ratio(clean_target, candidate_span)
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_result = SubtitleMatchResult(
                            start_time=window_cues[0].get("start_seconds", 0.0),
                            end_time=window_cues[-1].get("end_seconds", 0.0),
                            matched_text=window_raw.strip(),
                            similarity_score=round(score / 100.0, 4),
                            track_language=track_language,
                            is_auto_generated=is_auto_generated,
                        )

        return best_result

    def extract_audio_stream(
        self, url: str, job_id: str = "audio_stream", output_dir: Optional[Path] = None
    ) -> Path:
        """
        Extract audio stream to a local file for STT processing.
        If output_dir is provided, writes inside output_dir to enable caller cleanup.
        """
        clean_url = url.strip()
        if Path(clean_url).exists():
            return Path(clean_url)

        target_dir = output_dir if output_dir else Path(tempfile.mkdtemp(prefix=f"quest1_audio_{job_id}_"))
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{job_id}_audio.wav"

        try:
            opts = dict(self.ydl_opts)
            opts.update({
                "format": "bestaudio/best",
                "outtmpl": str(target_dir / f"{job_id}_audio.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }],
            })
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([clean_url])

            extracted = list(target_dir.glob(f"{job_id}_audio.*"))
            if extracted:
                return extracted[0]

            output_path.touch()
            return output_path
        except Exception as exc:
            logger.error("Audio extraction failed for %s: %s", clean_url, exc)
            raise IngestionError(f"Audio extraction failed for URL '{clean_url}': {exc}") from exc