import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
TranscriptsDisabled,
NoTranscriptFound,
VideoUnavailable,
)


class TranscriptUnavailableError(Exception):
    pass


VIDEO_ID_PATTERNS = [
    r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
    r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
    r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
]


def extract_video_id(url: str) -> str:
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    for pattern in VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a video ID from URL: {url}")


def get_transcript(video_id: str) -> list[dict] | None:
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        return [
            {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
            for snippet in fetched
        ]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        print(f"Unexpected error fetching transcript for {video_id}: {e}")
        return None


def get_title(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException:
        return f"YouTube video {video_id}"

    soup = BeautifulSoup(response.text, "html.parser")
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    return f"YouTube video {video_id}"