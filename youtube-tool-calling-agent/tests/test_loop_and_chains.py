"""Manual loop, the lab's fixed chain and recursive chain."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import calls
from yt_agent import build_fixed_chain, build_recursive_chain, collect_tool_calls, execute_tool, run_tool_loop
from yt_agent.offline import OfflineYouTubeModel

THREE_ROUNDS = "Show top 3 US trending videos with metadata and thumbnails"


def test_execute_tool_json_str_and_errors():
    msg = execute_tool({"name": "get_full_metadata", "args": {"url": "SmplAI00001"}, "id": "c1"})
    assert json.loads(msg.content)["title"] == "Tool Calling in 10 Minutes" and msg.tool_call_id == "c1"
    assert execute_tool({"name": "extract_video_id", "args": {"url": "https://youtu.be/SmplAI00001"}, "id": "c2"}).content == "SmplAI00001"
    assert execute_tool({"name": "nope", "args": {}, "id": "c3"}).content.startswith("Error: KeyError: \"unknown tool 'nope'")
    assert execute_tool({"name": "get_thumbnails", "args": {}, "id": "c4"}).content.startswith("Error: ValidationError")


def test_manual_loop_two_rounds(scripted_model):
    llm = scripted_model(calls(("extract_video_id", {"url": "https://youtu.be/SmplAI00001"})),
                         calls(("fetch_transcript", {"video_id": "SmplAI00001"})), AIMessage(content="It explains tool calling."))
    result = run_tool_loop(llm, "Summarize https://youtu.be/SmplAI00001")
    assert [type(m) for m in result.messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage, AIMessage]
    assert result.rounds == 2 and result.answer == "It explains tool calling."
    assert json.loads(result.tool_calls[1]["result"])["video_id"] == "SmplAI00001"


def test_manual_loop_parallel_and_max_steps(scripted_model):
    llm = scripted_model(calls(("get_full_metadata", {"url": "SmplAI00001"}), ("get_full_metadata", {"url": "SmplAI00002"})),
                         AIMessage(content="done"))
    assert len(run_tool_loop(llm, "two videos").tool_calls) == 2
    endless = scripted_model(*[calls(("extract_video_id", {"url": "SmplAI00001"}))] * 4)
    stopped = run_tool_loop(endless, "loop", max_steps=3)
    assert stopped.stopped and stopped.rounds == 3


def test_fixed_chain_fits_two_round_tasks():
    out = build_fixed_chain(OfflineYouTubeModel()).invoke({"query": "Summarize https://youtu.be/SmplAI00002"})
    assert out["complete"] and out["answer"].startswith("Summary")
    assert [c["name"] for c in collect_tool_calls(out["messages"])] == ["extract_video_id", "fetch_transcript"]


def test_fixed_chain_wastes_a_call_on_one_round_tasks():
    out = build_fixed_chain(OfflineYouTubeModel()).invoke({"query": "Show metadata for https://youtu.be/SmplAI00002"})
    answers = [m for m in out["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert out["complete"] and len(answers) == 2  # the same answer generated twice


def test_fixed_chain_cannot_finish_three_round_tasks():
    out = build_fixed_chain(OfflineYouTubeModel()).invoke({"query": THREE_ROUNDS})
    assert not out["complete"] and out["answer"] == ""


def test_recursive_chain_handles_any_number_of_rounds():
    messages = build_recursive_chain(OfflineYouTubeModel()).invoke({"query": THREE_ROUNDS})
    names = [c["name"] for c in collect_tool_calls(messages)]
    assert names == ["get_trending_videos"] + ["get_full_metadata"] * 3 + ["get_thumbnails"] * 3
    assert messages[-1].content.startswith("Top 3 trending in US")


def test_recursive_chain_has_a_depth_guard(scripted_model):
    endless = scripted_model(*[calls(("extract_video_id", {"url": "SmplAI00001"}))] * 10)
    messages = build_recursive_chain(endless, max_rounds=3).invoke({"query": "loop"})
    assert messages[-1].content == "Stopped after 3 tool rounds."
