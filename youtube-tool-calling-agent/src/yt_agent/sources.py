"""Where video data comes from: live YouTube, or a bundled sample catalogue.

Both sources expose the same small interface, so the tools, agents and tests don't
care which one is active:

- ``SampleSource`` reads ``data/sample_videos.json`` (made-up videos): no network,
  deterministic, used for demos, CI and screenshots.
- ``LiveSource`` uses ``yt-dlp`` (search, metadata, thumbnails, trending) and
  ``youtube-transcript-api`` (transcripts). Install with ``pip install -e ".[youtube]"``.
"""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from typing import Any, Optional, Protocol

VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([A-Za-z0-9_-]{11})")
BARE_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")

TRENDING_RETIRED = ("YouTube retired its Trending page in July 2025, so live trending lists are no longer "
                    "available. Use search_youtube instead, or the sample source for demos.")


class SourceError(Exception):
    """A lookup failed (unknown video, no transcript, network error, ...)."""


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Return the 11-character video ID from a URL (watch, youtu.be, embed, shorts, live) or a bare ID."""
    if not isinstance(url_or_id, str):
        return None
    text = url_or_id.strip()
    if BARE_ID_RE.fullmatch(text):
        return text
    match = VIDEO_ID_RE.search(text)
    return match.group(1) if match else None


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


class VideoSource(Protocol):
    name: str

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...
    def transcript(self, video_id: str, language: str = "en") -> str: ...
    def metadata(self, video_id: str) -> dict[str, Any]: ...
    def trending(self, region: str, limit: int = 10) -> list[dict[str, Any]]: ...
    def thumbnails(self, video_id: str) -> list[dict[str, Any]]: ...


# ----------------------------------------------------------------------------------
class SampleSource:
    """Bundled, made-up catalogue - deterministic and offline."""

    name = "sample"

    def __init__(self, data: Optional[dict] = None):
        if data is None:
            data = json.loads(resources.files("yt_agent").joinpath("data/sample_videos.json").read_text("utf-8"))
        self.videos = {v["video_id"]: v for v in data["videos"]}
        self.trending_by_region = {k.upper(): v for k, v in data["trending"].items()}

    def _video(self, video_id: str) -> dict:
        if video_id not in self.videos:
            raise SourceError(f"Video '{video_id}' not found in the sample catalogue.")
        return self.videos[video_id]

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored = []
        for v in self.videos.values():
            haystack = " ".join([v["title"], v["description"], " ".join(v["tags"]), v["channel"]]).lower()
            score = sum(1 for w in words if w in haystack)
            if score:
                scored.append((score, v["views"], v))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [_summary(v) for _, _, v in scored[:limit]]

    def transcript(self, video_id: str, language: str = "en") -> str:
        transcripts = self._video(video_id)["transcripts"]
        if language not in transcripts:
            raise SourceError(f"No '{language}' transcript for {video_id}. Available: {', '.join(transcripts)}.")
        return transcripts[language]

    def metadata(self, video_id: str) -> dict[str, Any]:
        v = self._video(video_id)
        return {"video_id": video_id, "title": v["title"], "channel": v["channel"], "views": v["views"],
                "duration": v["duration"], "likes": v["likes"], "comments": v["comments"],
                "published": v["published"], "chapters": v["chapters"], "url": watch_url(video_id)}

    def trending(self, region: str, limit: int = 10) -> list[dict[str, Any]]:
        ids = self.trending_by_region.get(region.upper())
        if ids is None:
            raise SourceError(f"No sample trending list for '{region}'. Available: {', '.join(self.trending_by_region)}.")
        return [_summary(self.videos[i]) for i in ids[:limit]]

    def thumbnails(self, video_id: str) -> list[dict[str, Any]]:
        return [{**t, "resolution": f"{t['width']}x{t['height']}"} for t in self._video(video_id)["thumbnails"]]


def _summary(v: dict) -> dict[str, Any]:
    return {"title": v["title"], "video_id": v["video_id"], "url": watch_url(v["video_id"]),
            "channel": v["channel"], "views": v["views"], "duration": v["duration"]}


# ----------------------------------------------------------------------------------
class LiveSource:
    """Real YouTube via yt-dlp and youtube-transcript-api (network required)."""

    name = "live"

    def __init__(self):
        try:
            import yt_dlp  # noqa: F401
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise SourceError('Live YouTube needs extra packages: pip install -e ".[youtube]"') from exc

    @staticmethod
    def _ydl(**opts):
        import yt_dlp

        return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, **opts})

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            with self._ydl(extract_flat=True) as ydl:
                info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        except Exception as exc:
            raise SourceError(f"Search failed: {exc}") from exc
        return [{"title": e.get("title"), "video_id": e.get("id"), "url": watch_url(e.get("id")),
                 "channel": e.get("channel") or e.get("uploader"), "views": e.get("view_count"),
                 "duration": e.get("duration")} for e in info.get("entries") or [] if e.get("id")]

    def transcript(self, video_id: str, language: str = "en") -> str:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        except Exception as exc:
            raise SourceError(f"No transcript for {video_id} ({language}): {type(exc).__name__}") from exc
        return " ".join(s.text for s in fetched.snippets)

    def metadata(self, video_id: str) -> dict[str, Any]:
        info = self._info(video_id)
        return {"video_id": video_id, "title": info.get("title"), "channel": info.get("uploader"),
                "views": info.get("view_count"), "duration": info.get("duration"), "likes": info.get("like_count"),
                "comments": info.get("comment_count"), "published": info.get("upload_date"),
                "chapters": info.get("chapters") or [], "url": watch_url(video_id)}

    def trending(self, region: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            with self._ydl(extract_flat=True, geo_bypass_country=region.upper()) as ydl:
                info = ydl.extract_info("https://www.youtube.com/feed/trending", download=False)
        except Exception as exc:
            raise SourceError(TRENDING_RETIRED) from exc
        entries = [e for e in info.get("entries") or [] if e.get("id")]
        if not entries:
            raise SourceError(TRENDING_RETIRED)
        return [{"title": e.get("title"), "video_id": e["id"], "url": watch_url(e["id"]),
                 "channel": e.get("uploader"), "views": e.get("view_count"), "duration": e.get("duration")}
                for e in entries[:limit]]

    def thumbnails(self, video_id: str) -> list[dict[str, Any]]:
        return [{"url": t["url"], "width": t.get("width"), "height": t.get("height"),
                 "resolution": f"{t.get('width')}x{t.get('height')}" if t.get("width") else None}
                for t in self._info(video_id).get("thumbnails") or [] if t.get("url")]

    def _info(self, video_id: str) -> dict:
        try:
            with self._ydl() as ydl:
                return ydl.extract_info(watch_url(video_id), download=False)
        except Exception as exc:
            raise SourceError(f"Could not load video {video_id}: {type(exc).__name__}") from exc


# ----------------------------------------------------------------------------------
_SOURCE: Optional[VideoSource] = None


def get_source() -> VideoSource:
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = make_source(os.getenv("VIDEO_SOURCE", "sample"))
    return _SOURCE


def make_source(name: str) -> VideoSource:
    name = name.lower()
    if name == "sample":
        return SampleSource()
    if name == "live":
        return LiveSource()
    raise ValueError(f"Unknown video source {name!r}. Use 'sample' or 'live'.")


def set_source(source: VideoSource | str) -> VideoSource:
    """Switch the source used by the tools (an instance, or 'sample' / 'live')."""
    global _SOURCE
    _SOURCE = make_source(source) if isinstance(source, str) else source
    return _SOURCE
