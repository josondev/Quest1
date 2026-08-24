import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.ingestion import StreamIngestionService, VideoMetadata


class TestStreamIngestionService:
    def test_sanitize_url_valid(self):
        """Ensure HTTP and HTTPS URLs pass sanitization."""
        svc = StreamIngestionService()
        assert svc.sanitize_url("https://www.youtube.com/watch?v=123") == "https://www.youtube.com/watch?v=123"
        assert svc.sanitize_url("http://example.com/video.mp4") == "http://example.com/video.mp4"

    def test_sanitize_url_invalid(self):
        """Ensure non-HTTP schemes (e.g. ftp, file, javascript) raise ValueError."""
        svc = StreamIngestionService()
        with pytest.raises(ValueError, match="Invalid or unsafe URL protocol"):
            svc.sanitize_url("ftp://example.com/video.mp4")

        with pytest.raises(ValueError, match="Invalid or unsafe URL protocol"):
            svc.sanitize_url("file:///etc/passwd")

    @patch("src.ingestion.yt_dlp.YoutubeDL")
    def test_probe_metadata_success(self, mock_ydl_cls, tmp_path):
        """Test metadata extraction with mocked yt-dlp dictionary response."""
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "duration": 184.5,
            "fps": 24.0,
            "width": 1920,
            "height": 1080,
            "title": "Sherlock Holmes Scene",
            "subtitles": {"en": [{"ext": "vtt", "url": "https://example.com/sub.vtt"}]},
            "automatic_captions": {}
        }

        svc = StreamIngestionService(temp_dir=tmp_path)
        meta = svc.probe_metadata("https://www.youtube.com/watch?v=test1234")

        assert isinstance(meta, VideoMetadata)
        assert meta.duration_seconds == 184.5
        assert meta.fps == 24.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.title == "Sherlock Holmes Scene"
        assert meta.has_subtitles is True
        assert "en" in meta.subtitles_info["manual"]

    @patch("src.ingestion.yt_dlp.YoutubeDL")
    def test_probe_metadata_error_handling(self, mock_ydl_cls, tmp_path):
        """Verify RuntimeError is raised when yt-dlp fails."""
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("Video unavailable")

        svc = StreamIngestionService(temp_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Failed to probe video stream metadata"):
            svc.probe_metadata("https://www.youtube.com/watch?v=badurl")

    @patch("src.ingestion.yt_dlp.YoutubeDL")
    def test_extract_audio_stream_success(self, mock_ydl_cls, tmp_path):
        """Verify audio stream download and path resolution."""
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"id": "vid123", "ext": "mp3"}

        expected_audio_file = tmp_path / "job_001_vid123.mp3"
        expected_audio_file.write_bytes(b"dummy audio data")

        svc = StreamIngestionService(temp_dir=tmp_path)
        output_file = svc.extract_audio_stream("https://www.youtube.com/watch?v=test", job_id="job_001")

        assert output_file.exists()
        assert output_file == expected_audio_file
