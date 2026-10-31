import argparse
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yt_dlp


DEFAULT_API_URL = (
    "https://quest1-api.ambitiousground-4a1b7fb4.centralindia."
    "azurecontainerapps.io"
)
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
YOUTUBE_DOMAINS = {"youtube.com", "youtu.be"}


def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in YOUTUBE_DOMAINS
        )
    )


def download_authorized_video(url: str, destination: Path) -> Path:
    if not is_youtube_url(url):
        raise ValueError("This local relay currently accepts HTTPS YouTube URLs only.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required locally to combine video and audio streams.")

    options = {
        "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/"
        "best[height<=360][ext=mp4]/best[height<=360]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "%(id)s.%(ext)s"),
        "max_filesize": MAX_UPLOAD_BYTES,
        "noplaylist": True,
        "ignoreconfig": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        if not info:
            raise RuntimeError("yt-dlp did not return video information.")

    candidates = [
        path
        for path in destination.iterdir()
        if path.is_file() and path.suffix not in {".part", ".ytdl"}
    ]
    if not candidates:
        raise RuntimeError("yt-dlp completed without producing a video file.")

    video_path = max(candidates, key=lambda path: path.stat().st_size)
    if video_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError(
            "Downloaded video exceeds the Quest1 upload limit of 250 MiB."
        )
    return video_path


def submit_video(api_url: str, video_path: Path, target_text: str) -> str:
    timeout = httpx.Timeout(300.0, connect=30.0)
    with httpx.Client(timeout=timeout) as client, video_path.open("rb") as video:
        response = client.post(
            f"{api_url.rstrip('/')}/api/v1/jobs/upload",
            data={"target_text": target_text},
            files={"file": (video_path.name, video, "video/mp4")},
        )
    response.raise_for_status()
    return response.json()["job_id"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch an authorized public YouTube video locally without cookies, "
            "then submit it to Quest1 for dialogue detection."
        )
    )
    parser.add_argument("--url", required=True, help="HTTPS YouTube video URL")
    parser.add_argument("--target-text", required=True, help="Dialogue to find")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("QUEST1_API_URL", DEFAULT_API_URL),
        help="Quest1 API base URL",
    )
    parser.add_argument(
        "--confirm-authorized",
        action="store_true",
        help="Confirm that you are authorized to download and process this video",
    )
    args = parser.parse_args()

    if not args.confirm_authorized:
        parser.error("Pass --confirm-authorized only if you have rights to use the video.")
    if not args.target_text.strip():
        parser.error("--target-text cannot be empty.")
    if not is_youtube_url(args.url):
        parser.error("--url must be an HTTPS YouTube URL.")

    try:
        with tempfile.TemporaryDirectory(prefix="quest1-youtube-") as temp_dir:
            print("Downloading the authorized video locally (no cookies are read or sent).")
            video_path = download_authorized_video(args.url, Path(temp_dir))
            print(f"Downloaded video: {video_path.stat().st_size / (1024 * 1024):.1f} MiB")
            job_id = submit_video(args.api_url, video_path, args.target_text.strip())
        print(f"Quest1 job submitted: {job_id}")
        return 0
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, yt_dlp.utils.DownloadError) as exc:
        print(f"Could not submit video: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
