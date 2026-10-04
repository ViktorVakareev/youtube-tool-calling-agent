"""Opt-in live checks.

    pytest -m live -v       # real LLM (GOOGLE_API_KEY / OPENAI_API_KEY) on the sample catalogue
    RUN_YOUTUBE_TESTS=1 pytest -m youtube -v     # real YouTube (network, pip install -e ".[youtube]")
"""

import os

import pytest

from yt_agent import collect_tool_calls, get_llm, run_tool_loop, set_source

PROVIDERS = [
    pytest.param("gemini", marks=pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")),
    pytest.param("openai", marks=pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
]


@pytest.mark.live
@pytest.mark.parametrize("provider", PROVIDERS)
def test_llm_summarizes_via_transcript(provider):
    result = run_tool_loop(get_llm(provider), "Summarize this video: https://www.youtube.com/watch?v=SmplAI00001")
    names = [c["name"] for c in result.tool_calls]
    assert "fetch_transcript" in names
    assert any(c["args"].get("video_id") == "SmplAI00001" for c in result.tool_calls if c["name"] == "fetch_transcript")
    assert result.answer


@pytest.mark.live
@pytest.mark.parametrize("provider", PROVIDERS)
def test_llm_gets_metadata_for_each_trending_video(provider):
    result = run_tool_loop(get_llm(provider), "Get the top 3 trending videos in India and their metadata")
    names = [c["name"] for c in result.tool_calls]
    assert names[0] == "get_trending_videos" and names.count("get_full_metadata") >= 3


@pytest.mark.youtube
@pytest.mark.skipif(not os.getenv("RUN_YOUTUBE_TESTS"), reason="set RUN_YOUTUBE_TESTS=1 to hit real YouTube")
def test_real_youtube_source():
    from yt_agent.tools import fetch_transcript, get_full_metadata

    set_source("live")
    meta = get_full_metadata.invoke({"url": "https://www.youtube.com/watch?v=aircAruvnKk"})
    assert "error" not in meta and meta["channel"]
    transcript = fetch_transcript.invoke({"video_id": "aircAruvnKk"})
    assert "error" in transcript or len(transcript["transcript"]) > 100  # captions can be blocked by YouTube
    assert collect_tool_calls([]) == []
