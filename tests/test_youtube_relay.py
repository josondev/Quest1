import pytest

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
