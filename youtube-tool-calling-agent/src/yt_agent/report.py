"""The lab's exercises as a pipeline: collect everything about one video, then ONE LLM call.

Contrast with the agent: here *your code* decides which tools run (always the same four),
and the model only writes the analysis. Cheaper and predictable when the steps are known.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .loop import text
from .tools import extract_video_id, fetch_transcript, get_full_metadata, get_thumbnails

REPORT_QUESTIONS = """Based on this information, please provide:
1. A concise summary of the video content (3-5 bullet points)
2. The main topics or themes discussed
3. The intended audience for this content
4. A brief analysis of why this video might be performing well (or not)"""


def collect_video_data(url: str, language: str = "en", max_transcript_chars: int = 3000) -> dict[str, Any]:
    """Exercises 2-5: video ID, metadata, transcript and thumbnails - errors recorded, not raised."""
    video_id = extract_video_id.invoke({"url": url})
    if video_id.startswith("Error"):
        return {"url": url, "error": video_id}
    metadata = get_full_metadata.invoke({"url": url})
    transcript = fetch_transcript.invoke({"video_id": video_id, "language": language, "max_chars": max_transcript_chars})
    thumbnails = get_thumbnails.invoke({"url": url})
    return {
        "url": url, "video_id": video_id,
        "metadata": metadata,
        "transcript": "" if "error" in transcript else transcript["transcript"],
        "transcript_truncated": bool(transcript.get("truncated")),
        "thumbnails": [] if isinstance(thumbnails, dict) else thumbnails,
        "errors": [r["error"] for r in (metadata, transcript, thumbnails) if isinstance(r, dict) and "error" in r],
    }


def build_report_prompt(data: dict[str, Any]) -> str:
    """The comprehensive prompt from the lab, built from collected data."""
    m = data.get("metadata") or {}
    duration = m.get("duration")
    duration_text = f"{duration // 60} min {duration % 60} s" if isinstance(duration, int) else "unknown"
    transcript = data.get("transcript") or "(no transcript available)"
    return f"""Please analyze this YouTube video and provide a comprehensive summary.

VIDEO TITLE: {m.get('title', 'unknown')}
CHANNEL: {m.get('channel', 'unknown')}
VIEWS: {m.get('views', 'unknown')}
DURATION: {duration_text}
LIKES: {m.get('likes', 'unknown')}
THUMBNAILS AVAILABLE: {len(data.get('thumbnails') or [])}

TRANSCRIPT EXCERPT:
{transcript}{" ... (truncated)" if data.get("transcript_truncated") else ""}

{REPORT_QUESTIONS}
"""


def video_report(url: str, llm: BaseChatModel, language: str = "en") -> dict[str, Any]:
    """Exercises 1-7 end to end: collect -> prompt -> single llm.invoke -> analysis."""
    data = collect_video_data(url, language)
    if "error" in data:
        return {**data, "analysis": None}
    prompt = build_report_prompt(data)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**data, "prompt": prompt, "analysis": text(response)}
