import pytest

from yt_agent import ask, build_agent, build_recursive_chain, final_answer, get_llm, run_tool_loop, tool_calls
from yt_agent.loop import collect_tool_calls
from yt_agent.offline import (OfflineYouTubeModel, analysis_from_prompt, detect_count, detect_language, detect_region,
                              search_topic, sentences)


@pytest.mark.parametrize("q, region", [("trending in India", "IN"), ("top 3 in the UK", "GB"), ("US trending", "US"),
                                       ("popular in Bulgaria", "BG"), ("trending", None)])
def test_detect_region(q, region):
    assert detect_region(q) == region


@pytest.mark.parametrize("q, n", [("top 3 videos", 3), ("top five", 5), ("Top 10", 10), ("trending", 3)])
def test_detect_count(q, n):
    assert detect_count(q) == n


def test_small_helpers():
    assert detect_language("summarize in Spanish") == "es" and detect_language("summarize") == "en"
    assert search_topic("Search for videos about python testing and summarize the top result") == "python testing"
    assert search_topic("find videos on sourdough bread") == "sourdough bread"
    assert sentences("One. Two! Three? Four.", 2) == ["One.", "Two!"]


def test_analysis_from_prompt():
    prompt = "VIDEO TITLE: T\nCHANNEL: C\nVIEWS: 1000\nLIKES: 50\nDURATION: 1 min 0 s\nTRANSCRIPT EXCERPT:\nAlpha beta. Gamma delta.\nBased on this information"
    out = analysis_from_prompt(prompt)
    assert 'Analysis of "T" by C' in out and "5.0% like-to-view ratio" in out and "- Alpha beta." in out


QUERIES = [
    ("Summarize https://www.youtube.com/watch?v=SmplAI00001", ["extract_video_id", "fetch_transcript"], "Summary"),
    ("Summarize https://youtu.be/SmplAI00001 in Spanish", ["extract_video_id", "fetch_transcript"], "Un modelo"),
    ("Show metadata and thumbnails for https://youtu.be/SmplCK00003", ["get_full_metadata", "get_thumbnails"], "4 thumbnails"),
    ("Show thumbnails for https://youtu.be/SmplCK00003", ["get_thumbnails"], "largest 1280x720"),
    ("How many views does https://youtu.be/SmplTR00005 have?", ["get_full_metadata"], "318,000 views"),
    ("Search for videos about python testing and summarize the top result", ["search_youtube", "fetch_transcript"], "pytest"),
    ("Find videos about quantum knitting", ["search_youtube"], "No matching videos"),
    ("Get top 3 youtube videos in India and their metadata", ["get_trending_videos"] + ["get_full_metadata"] * 3, "41,800 likes"),
    ("Top 2 trending in Bulgaria", ["get_trending_videos"], "Hidden Beaches"),
    ("Trending in France", ["get_trending_videos"], "couldn't complete"),
    ("Summarize https://youtu.be/NotThere123", ["extract_video_id", "fetch_transcript"], "not found"),
    ("hello", [], "no tool needed"),
]


@pytest.mark.parametrize("query, expected_tools, answer_part", QUERIES)
@pytest.mark.parametrize("engine", ["manual", "recursive", "agent"])
def test_offline_model_end_to_end(engine, query, expected_tools, answer_part):
    llm = OfflineYouTubeModel()
    if engine == "manual":
        result = run_tool_loop(llm, query)
        made, answer = result.tool_calls, result.answer
    elif engine == "recursive":
        messages = build_recursive_chain(llm).invoke({"query": query})
        made, answer = collect_tool_calls(messages), messages[-1].content
    else:
        response = ask(build_agent(llm), query)
        made, answer = tool_calls(response), final_answer(response)
    assert [c["name"] for c in made] == expected_tools
    assert answer_part in answer


def test_get_llm_providers(monkeypatch):
    assert isinstance(get_llm("offline"), OfflineYouTubeModel)
    with pytest.raises(ValueError):
        get_llm("nope")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert type(get_llm("gemini")).__name__ == "ChatGoogleGenerativeAI"
    assert type(get_llm("openai")).__name__ == "ChatOpenAI"


def test_tool_schemas_convert_for_gemini_and_openai():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    from yt_agent import TOOLS

    declared = [f.name for t in convert_to_genai_function_declarations(TOOLS) for f in (t.function_declarations or [])]
    assert declared == [t.name for t in TOOLS]
    assert convert_to_openai_tool(TOOLS[1])["function"]["parameters"]["required"] == ["video_id"]
