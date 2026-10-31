import argparse
import os
import shutil
import tempfile
import time
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
VIDEO_EXTENSIONS = {
    ".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".webm",
}


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


def _is_cookie_auth_error(error: yt_dlp.utils.DownloadError) -> bool:
    message = str(error).casefold()
    return any(
        phrase in message
        for phrase in ("sign in to confirm", "not a bot", "cookies are no longer valid")
    )


def download_authorized_video(
    url: str,
    destination: Path,
    cookies_file: Path | None = None,
) -> Path:
    if not is_youtube_url(url):
        raise ValueError("This local relay currently accepts HTTPS YouTube URLs only.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required locally to combine video and audio streams.")
    if cookies_file is not None:
        cookies_file = cookies_file.expanduser().resolve()
        if not cookies_file.is_file():
            raise FileNotFoundError(f"Cookie file not found: {cookies_file}")

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

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if not info:
                raise RuntimeError("yt-dlp did not return video information.")
    except yt_dlp.utils.DownloadError as exc:
        if cookies_file is None or not _is_cookie_auth_error(exc):
            raise

        print("Anonymous YouTube access was rejected; retrying with the specified local cookie file.")
        options["cookiefile"] = str(cookies_file)
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if not info:
                raise RuntimeError("yt-dlp did not return video information.")

    candidates = [
        path
        for path in destination.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
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


def wait_for_job(api_url: str, job_id: str, timeout_seconds: int = 1800) -> dict:
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=30.0) as client:
        while time.monotonic() < deadline:
            response = client.get(
                f"{api_url.rstrip('/')}/api/v1/jobs/{job_id}"
            )
            response.raise_for_status()
            result = response.json()
            if result["status"] in {
                "completed", "failed", "uncertain", "ambiguous"
            }:
                return result
            time.sleep(5)
    raise TimeoutError(f"Quest1 job {job_id} did not finish in time.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch an authorized YouTube video locally, optionally using a "
            "local cookie file as a fallback, then submit it to Quest1."
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
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help=(
            "Optional local YouTube cookies file. Used only as a fallback if "
            "anonymous download is rejected; it is never uploaded to Quest1."
        ),
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
            print("Downloading the authorized video locally; any cookies stay on this machine.")
            video_path = download_authorized_video(
                args.url,
                Path(temp_dir),
                args.cookies_file,
            )
            print(f"Downloaded video: {video_path.stat().st_size / (1024 * 1024):.1f} MiB")
            job_id = submit_video(args.api_url, video_path, args.target_text.strip())
        print(f"Quest1 job submitted: {job_id}")
        result = wait_for_job(args.api_url, job_id)
        print(f"Status: {result['status']}")
        if result.get("formatted_timestamp"):
            print(f"Timestamp: {result['formatted_timestamp']}")
        if result.get("extracted_text"):
            print(f"Matched dialogue: {result['extracted_text']}")
        if result.get("error_message"):
            print(f"Quest1 error: {result['error_message']}")
        return 1 if result["status"] == "failed" else 0
    except (
        httpx.HTTPError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
        yt_dlp.utils.DownloadError,
    ) as exc:
        print(f"Could not submit video: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
