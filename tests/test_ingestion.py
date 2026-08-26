from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion import IngestionError, StreamIngestionService


class TestStreamIngestionService:

    # 1. Text Normalization Contract
    def test_normalize_text_contract(self):
        svc = StreamIngestionService()
        assert svc._normalize_text("It's UPPERCASE!") == "its uppercase"
        assert svc._normalize_text("My   mind   rebels.") == "my mind rebels"
        assert svc._normalize_text("STAGNATION") == "stagnation"

    # 2. VTT Timestamp Parsing
    def test_parse_vtt_timestamp(self):
        svc = StreamIngestionService()
        assert svc._parse_vtt_timestamp("00:01:05.432") == 65.432
        assert svc._parse_vtt_timestamp("01:00.500") == 60.5
        assert svc._parse_vtt_timestamp("01:02:03,100") == 3723.1

    # 3. VTT File Parsing
    def test_vtt_file_parsing(self, tmp_path):
        svc = StreamIngestionService()
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("""WEBVTT

1
00:00:01.000 --> 00:00:04.000
My mind

2
00:00:04.100 --> 00:00:07.000
rebels at <i>stagnation</i>.
""", encoding="utf-8")

        cues = svc._parse_vtt_file(vtt_file)
        assert len(cues) == 2
        assert cues[0]["start_seconds"] == 1.0
        assert cues[0]["end_seconds"] == 4.0
        assert cues[0]["text"] == "My mind"
        assert cues[1]["text"] == "rebels at stagnation."

    # 4. Multi-Cue Match Across 3 Cues
    def test_match_phrase_across_3_cues(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 1.0, "end_seconds": 2.0, "text": "My mind"},
            {"start_seconds": 2.1, "end_seconds": 3.0, "text": "rebels at"},
            {"start_seconds": 3.1, "end_seconds": 4.0, "text": "stagnation."},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="My mind rebels at stagnation",
            cues=cues,
            similarity_threshold=85.0,
            max_gap_seconds=2.0,
            track_language="en",
            is_auto_generated=False,
        )

        assert result is not None
        assert result.start_time == 1.0
        assert result.end_time == 4.0
        assert result.similarity_score >= 0.85
        assert result.matched_text == "My mind rebels at stagnation."
        assert result.is_auto_generated is False

    # 5. Subtitle Embedded Inside Longer Cue
    def test_match_phrase_embedded_inside_longer_cue(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 10.0, "end_seconds": 15.0, "text": "We really need to leave now before dark."},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="leave now",
            cues=cues,
            similarity_threshold=85.0,
            max_gap_seconds=2.0,
        )

        assert result is not None
        assert result.start_time == 10.0
        assert result.end_time == 15.0
        assert result.similarity_score >= 0.85

    # 6. Short Prefix Cue Rejection
    def test_match_phrase_short_prefix_rejected(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 1.0, "end_seconds": 2.0, "text": "My mind"},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="My mind rebels at stagnation",
            cues=cues,
            similarity_threshold=85.0,
            max_gap_seconds=2.0,
        )

        assert result is None

    # 7. Excessive Temporal Gap Rejection
    def test_match_phrase_excessive_temporal_gap_rejected(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 1.0, "end_seconds": 2.0, "text": "My mind"},
            {"start_seconds": 10.0, "end_seconds": 11.0, "text": "rebels at stagnation"},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="My mind rebels at stagnation",
            cues=cues,
            similarity_threshold=85.0,
            max_gap_seconds=2.0,
        )

        assert result is None

    # 8. Fuzzy Matching with Typo
    def test_match_phrase_fuzzy_typo_match(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 1.0, "end_seconds": 4.0, "text": "My mind rebels at stagnaton."},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="My mind rebels at stagnation",
            cues=cues,
            similarity_threshold=80.0,
            max_gap_seconds=2.0,
        )

        assert result is not None
        assert result.similarity_score >= 0.80

    # 9. No Match Return
    def test_match_phrase_no_match(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 1.0, "end_seconds": 4.0, "text": "The quick brown fox jumps over the lazy dog."},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="My mind rebels at stagnation",
            cues=cues,
            similarity_threshold=85.0,
        )

        assert result is None

    # 10. Embedded Subtitle Match Success
    @patch.object(StreamIngestionService, "_download_subtitle_file")
    def test_probe_embedded_subtitles_match_uses_download_file_contract(self, mock_download_sub, tmp_path):
        vtt_file = tmp_path / "subs.vtt"
        vtt_file.write_text("""WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world
""", encoding="utf-8")

        mock_download_sub.return_value = (vtt_file, False, "en")

        service = StreamIngestionService()
        result = service.probe_embedded_subtitles_match("https://youtube.com/watch?v=test", "Hello world")

        assert result is not None
        assert result.matched_text == "Hello world"
        mock_download_sub.assert_called_once()

    # 11. Embedded Subtitle Match Returns None When No Subs Found
    @patch.object(StreamIngestionService, "_download_subtitle_file")
    def test_probe_embedded_subtitles_match_no_subs(self, mock_download_sub):
        mock_download_sub.return_value = (None, False, "en")

        service = StreamIngestionService()
        result = service.probe_embedded_subtitles_match("https://youtube.com/watch?v=test", "Hello world")

        assert result is None

    # 12. Manual Over Auto Subtitle Priority
    @patch("yt_dlp.YoutubeDL")
    def test_fetch_subtitles_manual_over_auto_priority(self, mock_ydl_cls, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "subtitles": {"en": [{"ext": "vtt"}]},
            "automatic_captions": {"en": [{"ext": "vtt"}]},
        }

        service = StreamIngestionService()
        def fake_download(urls):
            vtt = tmp_path / "sub_test.en.vtt"
            vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nManual text")

        mock_ydl.download.side_effect = fake_download

        res = service._download_subtitle_file("https://youtube.com/watch?v=test", target_dir=tmp_path)
        assert res[1] is False

    # 13. Auto Caption Fallback
    @patch("yt_dlp.YoutubeDL")
    def test_fetch_subtitles_auto_caption_fallback(self, mock_ydl_cls, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "subtitles": {},
            "automatic_captions": {"en": [{"ext": "vtt"}]},
        }

        service = StreamIngestionService()
        def fake_download(urls):
            vtt = tmp_path / "sub_test.en.vtt"
            vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nAuto text")

        mock_ydl.download.side_effect = fake_download

        res = service._download_subtitle_file("https://youtube.com/watch?v=test", target_dir=tmp_path)
        assert res[1] is True

    # 14. Local File Probing via OpenCV
    @patch("cv2.VideoCapture")
    def test_probe_metadata_local_file_opencv(self, mock_cap_cls, tmp_path):
        local_file = tmp_path / "test_video.mp4"
        local_file.touch()

        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else (900.0 if prop == 7 else 0)

        service = StreamIngestionService()
        meta = service.probe_metadata(str(local_file))

        assert meta.is_local is True
        assert meta.url == str(local_file)
        assert meta.fps == 30.0
        assert meta.duration_seconds == 30.0
        mock_cap.release.assert_called_once()

    # 15. Unopenable Local Video Probing Rejection
    @patch("cv2.VideoCapture")
    def test_probe_metadata_local_file_unopenable_raises_error(self, mock_cap_cls, tmp_path):
        local_file = tmp_path / "invalid_video.mp4"
        local_file.touch()

        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = False

        service = StreamIngestionService()
        with pytest.raises(IngestionError, match="Unable to open local video file"):
            service.probe_metadata(str(local_file))
        mock_cap.release.assert_called_once()

    # 16. Remote Video Probing via yt-dlp
    @patch("yt_dlp.YoutubeDL")
    def test_probe_metadata_remote_success(self, mock_ydl_cls):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "duration": 120.0,
            "fps": 30.0,
            "subtitles": {"en": [{"ext": "vtt"}]},
            "formats": [{"vcodec": "h264", "acodec": "aac", "url": "https://googlevideo.com/stream"}],
        }

        service = StreamIngestionService()
        meta = service.probe_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert meta.is_local is False
        assert meta.duration_seconds == 120.0
        assert meta.has_subtitles is True
        assert meta.stream_path == "https://googlevideo.com/stream"

    # 17. Remote Video Probing Failure
    @patch("yt_dlp.YoutubeDL")
    def test_probe_metadata_failure_raises_ingestion_error(self, mock_ydl_cls):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("Network blocked")

        service = StreamIngestionService()
        with pytest.raises(IngestionError, match="Probing metadata failed"):
            service.probe_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # 18. Local Audio Extraction Shortcut
    def test_extract_audio_stream_local(self, tmp_path):
        local_audio = tmp_path / "local_audio.wav"
        local_audio.touch()

        service = StreamIngestionService()
        res_path = service.extract_audio_stream(str(local_audio))
        assert res_path == local_audio

    # 19. Remote Audio Extraction Success
    @patch("yt_dlp.YoutubeDL")
    def test_extract_audio_stream_remote(self, mock_ydl_cls, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        def fake_download(urls):
            out_file = tmp_path / "audio_job1_audio.wav"
            out_file.touch()

        mock_ydl.download.side_effect = fake_download

        service = StreamIngestionService()
        audio_file = service.extract_audio_stream("https://youtube.com/watch?v=test", job_id="job1", output_dir=tmp_path)
        assert audio_file.exists()

    # 20. Remote Audio Extraction Failure
    @patch("yt_dlp.YoutubeDL")
    def test_extract_audio_stream_failure(self, mock_ydl_cls, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.download.side_effect = Exception("FFmpeg missing")

        service = StreamIngestionService()
        with pytest.raises(IngestionError, match="Audio extraction failed"):
            service.extract_audio_stream("https://youtube.com/watch?v=test", job_id="job1", output_dir=tmp_path)

    # 21. Repeated Whitespace Normalization Match
    def test_match_phrase_normalizes_repeated_whitespace(self):
        service = StreamIngestionService()
        cues = [
            {"start_seconds": 1.0, "end_seconds": 4.0, "text": "My    mind    rebels."},
        ]

        result = service._match_phrase_in_cues(
            target_phrase="My mind rebels",
            cues=cues,
            similarity_threshold=85.0,
        )

        assert result is not None
        assert result.similarity_score >= 0.85