"""``OfflineYouTubeModel`` - a deterministic, rule-based stand-in for an LLM.

It is **not an LLM**. It speaks the real tool-calling protocol (``AIMessage.tool_calls``
in, ``ToolMessage`` results out), so the manual loop, both LCEL chains, ``create_agent``
and the report pipeline all run for real without an API key. Its "summaries" are
extractive (the first sentences of the transcript), and it only understands simple
requests:

- "summarize <url>"                       -> extract_video_id -> fetch_transcript
- "metadata / stats / details for <url>"   -> get_full_metadata
- "thumbnails for <url>"                   -> get_thumbnails
- "search <topic>" (+ "summarize the top") -> search_youtube (-> fetch_transcript)
- "top N trending in <country>" (+ "metadata", "thumbnails")
                                           -> get_trending_videos -> get_full_metadata xN -> get_thumbnails xN
- a report prompt ("VIDEO TITLE: ...")      -> a templated analysis, no tools
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

URL_RE = re.compile(r"https?://\S+")
REGIONS = {
    # no bare "in" / "de": they are ordinary words ("top 3 in the UK")
    "us": "US", "usa": "US", "united states": "US", "america": "US", "india": "IN",
    "uk": "GB", "britain": "GB", "united kingdom": "GB", "england": "GB", "gb": "GB",
    "bulgaria": "BG", "bg": "BG", "germany": "DE", "france": "FR", "spain": "ES", "italy": "IT",
    "japan": "JP", "canada": "CA", "brazil": "BR", "australia": "AU",
}
LANGUAGES = {"spanish": "es", "english": "en", "german": "de", "french": "fr", "bulgarian": "bg"}
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def _load(content: Any) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (TypeError, ValueError, SyntaxError):
            continue
    return content


def sentences(text: str, n: int = 3) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    return parts[:n]


def detect_region(query: str) -> Optional[str]:
    lowered = query.lower()
    for name in sorted(REGIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return REGIONS[name]
    return None


def detect_count(query: str, default: int = 3) -> int:
    m = re.search(r"\btop\s+(\d+|one|two|three|four|five)\b", query, re.IGNORECASE)
    if not m:
        return default
    token = m.group(1).lower()
    return int(token) if token.isdigit() else _WORD_NUMBERS[token]


def detect_language(query: str) -> str:
    for word, code in LANGUAGES.items():
        if word in query.lower():
            return code
    return "en"


def search_topic(query: str) -> str:
    m = re.search(r"(?:search(?: youtube)?(?: for)?|find(?: videos)?(?: about| on)?|videos about|look for)\s+(.+?)"
                  r"(?:\s+and\s+summari[sz]e.*|\s*[?.!]?$)", query, re.IGNORECASE)
    topic = (m.group(1) if m else query).strip(" '\"")
    return re.sub(r"^(?:videos?|youtube videos?)\s+(?:about|on)\s+", "", topic, flags=re.IGNORECASE)


class OfflineYouTubeModel(BaseChatModel):
    """Rule-based chat model driving the YouTube tools through real tool calls."""

    @property
    def _llm_type(self) -> str:
        return "offline-youtube-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineYouTubeModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        query = str(messages[start].content)
        turn = messages[start:]
        ids = {c["id"]: c["name"] for m in turn if isinstance(m, AIMessage) for c in m.tool_calls}
        results: dict[str, list[Any]] = {}
        for m in turn:
            if isinstance(m, ToolMessage):
                results.setdefault(m.name or ids.get(m.tool_call_id, ""), []).append(_load(m.content))
        return ChatResult(generations=[ChatGeneration(message=self._next(query, results))])

    # -------------------------------------------------------------------- planning
    def _next(self, query: str, r: dict[str, list[Any]]) -> AIMessage:
        if "VIDEO TITLE:" in query:
            return AIMessage(content=analysis_from_prompt(query))

        q = query.lower()
        url_match = URL_RE.search(query)
        errors = [x["error"] for xs in r.values() for x in xs if isinstance(x, dict) and "error" in x]
        errors += [x for xs in r.values() for x in xs if isinstance(x, str) and x.startswith("Error")]

        is_search = bool(re.search(r"\b(search|find|look for|videos about)\b", q))
        is_trending = "trending" in q or "popular in" in q or (bool(re.search(r"\btop\b", q)) and detect_region(query))
        if is_trending and not is_search and not url_match:
            return self._trending(query, r, errors)
        if url_match:
            url = url_match.group(0).rstrip(".,)")
            if errors:
                return AIMessage(content=f"I couldn't complete that: {errors[0]}")
            if "thumbnail" in q and "metadata" not in q:
                if "get_thumbnails" not in r:
                    return AIMessage(content="", tool_calls=[_call("get_thumbnails", url=url)])
                return AIMessage(content=_thumbs_answer(r["get_thumbnails"][0]))
            if any(w in q for w in ("metadata", "stats", "statistics", "views", "details", "how long", "likes")):
                calls = [] if "get_full_metadata" in r else [_call("get_full_metadata", url=url)]
                if "thumbnail" in q and "get_thumbnails" not in r:
                    calls.append(_call("get_thumbnails", url=url))
                if calls:
                    return AIMessage(content="", tool_calls=calls)
                answer = _meta_answer(r["get_full_metadata"][0])
                if "get_thumbnails" in r:
                    answer += " " + _thumbs_answer(r["get_thumbnails"][0])
                return AIMessage(content=answer)
            # default with a URL: summarize
            if "extract_video_id" not in r:
                return AIMessage(content="", tool_calls=[_call("extract_video_id", url=url)])
            if "fetch_transcript" not in r:
                return AIMessage(content="", tool_calls=[_call("fetch_transcript", video_id=r["extract_video_id"][0],
                                                               language=detect_language(query))])
            return AIMessage(content=_summary_answer(r["fetch_transcript"][0]))

        if is_search:
            if "search_youtube" not in r:
                return AIMessage(content="", tool_calls=[_call("search_youtube", query=search_topic(query), max_results=3)])
            hits = r["search_youtube"][0]
            if errors or not hits:
                return AIMessage(content=f"No results: {errors[0]}" if errors else "No matching videos found.")
            if re.search(r"summari[sz]e", q):
                if "fetch_transcript" not in r:
                    return AIMessage(content="", tool_calls=[_call("fetch_transcript", video_id=hits[0]["video_id"])])
                return AIMessage(content=f"Top result: \"{hits[0]['title']}\" ({hits[0]['channel']}). "
                                         + _summary_answer(r["fetch_transcript"][0]))
            listing = "\n".join(f"{i}. {h['title']} - {h['channel']} ({h['url']})" for i, h in enumerate(hits, 1))
            return AIMessage(content=f"Found {len(hits)} video{'s' if len(hits) != 1 else ''}:\n{listing}")

        return AIMessage(content="Offline demo model: no tool needed. Try \"Summarize <YouTube URL>\", "
                                 "\"Search for videos about python testing\" or \"Top 3 trending videos in India with metadata\".")

    def _trending(self, query: str, r: dict[str, list[Any]], errors: list[str]) -> AIMessage:
        if "get_trending_videos" not in r:
            return AIMessage(content="", tool_calls=[_call("get_trending_videos", region_code=detect_region(query) or "US")])
        if errors:
            return AIMessage(content=f"I couldn't complete that: {errors[0]}")
        top = r["get_trending_videos"][0][:detect_count(query)]
        q = query.lower()
        if "metadata" in q and "get_full_metadata" not in r:  # parallel calls, one per video
            return AIMessage(content="", tool_calls=[_call("get_full_metadata", url=v["url"]) for v in top])
        if "thumbnail" in q and "get_thumbnails" not in r:
            return AIMessage(content="", tool_calls=[_call("get_thumbnails", url=v["url"]) for v in top])
        metas = r.get("get_full_metadata") or []
        thumbs = r.get("get_thumbnails") or []
        lines = []
        for i, v in enumerate(top):
            m = metas[i] if i < len(metas) else v
            line = (f"{i + 1}. {m['title']} - {m['channel']}, {m['views']:,} views, "
                    f"{_duration(m.get('duration'))}")
            if "likes" in m:
                line += f", {m['likes']:,} likes"
            if i < len(thumbs) and isinstance(thumbs[i], list) and thumbs[i]:
                best = max(thumbs[i], key=lambda t: (t.get("width") or 0))
                line += f", thumbnail {best['resolution']}"
            lines.append(line)
        region = detect_region(query) or "US"
        return AIMessage(content=f"Top {len(top)} trending in {region}:\n" + "\n".join(lines))


# ------------------------------------------------------------------------ answers
def _duration(seconds: Any) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}" if isinstance(seconds, int) else "?"


def _summary_answer(transcript: Any) -> str:
    if not isinstance(transcript, dict) or "transcript" not in transcript:
        return f"I couldn't get the transcript: {transcript}"
    points = sentences(transcript["transcript"], 3)
    return "Summary (extractive, offline model):\n" + "\n".join(f"- {p}" for p in points)


def _meta_answer(m: dict) -> str:
    chapters = ", ".join(c["title"] for c in m.get("chapters") or []) or "none"
    return (f"\"{m['title']}\" by {m['channel']}: {m['views']:,} views, {m['likes']:,} likes, "
            f"{m['comments']:,} comments, length {_duration(m['duration'])}. Chapters: {chapters}.")


def _thumbs_answer(thumbs: Any) -> str:
    if not isinstance(thumbs, list) or not thumbs:
        return "No thumbnails found."
    best = max(thumbs, key=lambda t: (t.get("width") or 0))
    return f"{len(thumbs)} thumbnails; largest {best['resolution']}: {best['url']}"


def analysis_from_prompt(prompt: str) -> str:
    """Templated 'analysis' for the report prompt - stands in for the single LLM call."""
    field = lambda name: (re.search(rf"^{name}:\s*(.*)$", prompt, re.MULTILINE) or [None, "unknown"])[1].strip()  # noqa: E731
    excerpt = prompt.split("TRANSCRIPT EXCERPT:", 1)[-1].split("Based on this information", 1)[0].strip()
    points = sentences(excerpt, 4)
    stop = {"their", "there", "which", "about", "would", "these", "every", "without", "never", "before", "after",
            "what", "with", "your", "from", "that", "this", "into", "home", "minutes", "minute"}
    title_words = [w for w in re.findall(r"[a-z]{4,}", field("VIDEO TITLE").lower()) if w not in stop]
    words = [w for w in re.findall(r"[a-z]{5,}", excerpt.lower()) if w not in stop]
    frequent = [w for w in sorted(set(words), key=lambda w: (-words.count(w), words.index(w))) if words.count(w) > 1]
    top = list(dict.fromkeys(title_words + frequent))[:4]
    views, likes = field("VIEWS"), field("LIKES")
    ratio = ""
    if views.isdigit() and likes.isdigit() and int(views):
        ratio = f" with a {int(likes) / int(views):.1%} like-to-view ratio"
    return (
        f"Analysis of \"{field('VIDEO TITLE')}\" by {field('CHANNEL')} (offline demo model, templated):\n\n"
        "1. Summary:\n" + "\n".join(f"   - {p}" for p in points) + "\n"
        f"2. Main topics: {', '.join(top) or 'n/a'}\n"
        f"3. Audience: viewers interested in {field('VIDEO TITLE').lower()}; beginner-friendly tone.\n"
        f"4. Performance: {views} views{ratio}; length {field('DURATION')}."
    )
