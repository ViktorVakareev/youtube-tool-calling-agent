from yt_agent.tools import (TOOL_MAPPING, TOOLS, extract_video_id, fetch_transcript, get_full_metadata, get_thumbnails,
                            get_trending_videos, search_youtube)


def test_extract_video_id_tool():
    assert extract_video_id.invoke({"url": "https://youtu.be/SmplAI00001"}) == "SmplAI00001"
    assert extract_video_id.invoke({"url": "nope"}) == "Error: Invalid YouTube URL"
    assert extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A") == "hfIUstzHs9A"  # the lab's .run() style


def test_fetch_transcript_accepts_id_or_url_and_truncates():
    full = fetch_transcript.invoke({"video_id": "SmplAI00001"})
    assert full["truncated"] is False and full["language"] == "en"
    short = fetch_transcript.invoke({"video_id": "https://youtu.be/SmplAI00001", "max_chars": 20})
    assert len(short["transcript"]) == 20 and short["truncated"] is True


def test_other_tools_happy_paths():
    assert search_youtube.invoke({"query": "sourdough bread"})[0]["title"] == "Perfect Sourdough at Home"
    assert get_full_metadata.invoke({"url": "https://youtu.be/SmplTR00005"})["channel"] == "Wander Notes"
    assert len(get_trending_videos.invoke({"region_code": "gb", "max_results": 2})) == 2
    assert len(get_thumbnails.invoke({"url": "SmplPY00006"})) == 4


def test_errors_are_returned_not_raised():
    assert "not found" in fetch_transcript.invoke({"video_id": "NotThere123"})["error"]
    assert "Could not find a YouTube video ID" in fetch_transcript.invoke({"video_id": "abc"})["error"]
    assert "not found" in get_full_metadata.invoke({"url": "https://youtu.be/NotThere123"})["error"]
    assert "Could not find" in get_thumbnails.invoke({"url": "https://example.com"})["error"]
    assert "Available" in get_trending_videos.invoke({"region_code": "ZZ"})["error"]
    assert "Could not find" in get_full_metadata.invoke({"url": "x"})["error"]


def test_search_errors_from_source(sample_source, monkeypatch):
    from yt_agent.sources import SourceError

    def boom(*a, **k):
        raise SourceError("search down")
    monkeypatch.setattr(sample_source, "search", boom)
    assert search_youtube.invoke({"query": "x"}) == {"error": "search down"}


def test_schemas_the_llm_sees():
    assert [t.name for t in TOOLS] == ["extract_video_id", "fetch_transcript", "search_youtube", "get_full_metadata",
                                       "get_trending_videos", "get_thumbnails"]
    assert set(fetch_transcript.args) == {"video_id", "language", "max_chars"}
    assert fetch_transcript.args["language"]["default"] == "en"
    assert TOOL_MAPPING["get_thumbnails"] is get_thumbnails
    for t in TOOLS:
        assert len(t.description) > 40
