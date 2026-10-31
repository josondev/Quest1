import pytest

from tools import submit_youtube_url as relay
from tools.submit_youtube_url import is_youtube_url


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=example",
        "https://www.youtube.com/watch?v=example",
        "https://youtu.be/example",
        "https://m.youtube.com/watch?v=example",
    ],
)
def test_accepts_youtube_https_urls(url):
    assert is_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://youtube.com/watch?v=example",
        "https://youtube.com.example.org/watch?v=example",
        "https://instagram.com/reel/example",
        "file:///tmp/video.mp4",
    ],
)
def test_rejects_non_youtube_urls(url):
    assert not is_youtube_url(url)


def test_cookie_file_is_only_used_after_local_bot_challenge(
    tmp_path, monkeypatch
):
    cookies_file = tmp_path / "youtube-cookies.txt"
    cookies_file.write_text("cookie data", encoding="utf-8")
    options_seen = []

    class FakeDownloader:
        def __init__(self, options):
            self.options = dict(options)
            options_seen.append(self.options)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, _url, download):
            assert download is True
            if not self.options.get("cookiefile"):
                raise relay.yt_dlp.utils.DownloadError(
                    "Sign in to confirm you're not a bot"
                )
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"video")
            return {"id": "test-video"}

    monkeypatch.setattr(relay.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(relay.yt_dlp, "YoutubeDL", FakeDownloader)

    result = relay.download_authorized_video(
        "https://youtu.be/example",
        tmp_path,
        cookies_file=cookies_file,
    )

    assert result.read_bytes() == b"video"
    assert "cookiefile" not in options_seen[0]
    assert options_seen[1]["cookiefile"] == str(cookies_file.resolve())
