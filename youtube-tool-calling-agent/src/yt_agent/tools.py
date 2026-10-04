"""The six YouTube tools the LLM can call.

Same names and jobs as in the lab, with consistent return shapes: every tool returns
its data, or ``{"error": "..."}`` - it never raises, so one failed call can't crash
the agent. Transcripts are truncated (``max_chars``) to keep token costs predictable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .sources import SourceError, get_source
from .sources import extract_video_id as _extract_id


def _id_or_error(url_or_id: str) -> str | dict:
    video_id = _extract_id(url_or_id)
    return video_id if video_id else {"error": f"Could not find a YouTube video ID in {url_or_id!r}."}


@tool
def extract_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from a URL (watch, youtu.be, embed or shorts link).

    Returns the ID, e.g. "dQw4w9WgXcQ", or a string starting with "Error:".
    """
    video_id = _extract_id(url)
    return video_id or "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en", max_chars: int = 4000) -> Any:
    """Fetch the spoken transcript of a YouTube video - the content to summarize or answer questions about.

    Args:
        video_id: the 11-character video ID (use extract_video_id on a URL first).
        language: transcript language code, e.g. "en", "es".
        max_chars: maximum characters returned (long transcripts are truncated).

    Returns:
        {"video_id", "language", "transcript", "truncated"} or {"error": "..."}.
    """
    vid = _id_or_error(video_id)
    if isinstance(vid, dict):
        return vid
    try:
        text = get_source().transcript(vid, language)
    except SourceError as exc:
        return {"error": str(exc)}
    return {"video_id": vid, "language": language, "transcript": text[:max_chars], "truncated": len(text) > max_chars}


@tool
def search_youtube(query: str, max_results: int = 5) -> Any:
    """Search YouTube for videos about a topic.

    Returns a list of {"title", "video_id", "url", "channel", "views", "duration"} or {"error": "..."}.
    """
    try:
        return get_source().search(query, max_results)
    except SourceError as exc:
        return {"error": str(exc)}


@tool
def get_full_metadata(url: str) -> Any:
    """Get a video's details from its URL or ID: title, channel, views, duration (seconds), likes,
    comments, publish date and chapters.

    Returns a dict, or {"error": "..."}.
    """
    vid = _id_or_error(url)
    if isinstance(vid, dict):
        return vid
    try:
        return get_source().metadata(vid)
    except SourceError as exc:
        return {"error": str(exc)}


@tool
def get_trending_videos(region_code: str, max_results: int = 10) -> Any:
    """List trending videos for a country, by 2-letter region code ("US", "IN", "GB", ...).

    Returns a list of {"title", "video_id", "url", "channel", "views", "duration"} or {"error": "..."}.
    Note: YouTube retired its Trending page in July 2025; the live source reports that as an error.
    """
    try:
        return get_source().trending(region_code, max_results)
    except SourceError as exc:
        return {"error": str(exc)}


@tool
def get_thumbnails(url: str) -> Any:
    """List the thumbnail images of a video (URL or ID): each with url, width, height and resolution.

    Returns a list, or {"error": "..."}.
    """
    vid = _id_or_error(url)
    if isinstance(vid, dict):
        return vid
    try:
        return get_source().thumbnails(vid)
    except SourceError as exc:
        return {"error": str(exc)}


TOOLS = [extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_trending_videos, get_thumbnails]
TOOL_MAPPING = {t.name: t for t in TOOLS}
