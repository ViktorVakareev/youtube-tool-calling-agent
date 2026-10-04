"""YouTube tool-calling agent: the LangChain tool loop by hand, as LCEL chains, and with create_agent."""

from .agent import SYSTEM_PROMPT, ask, build_agent, final_answer, get_llm, tool_calls
from .chains import build_fixed_chain, build_recursive_chain
from .loop import LoopResult, collect_tool_calls, execute_tool, run_tool_loop
from .report import build_report_prompt, collect_video_data, video_report
from .sources import LiveSource, SampleSource, SourceError, get_source, set_source
from .tools import (TOOL_MAPPING, TOOLS, extract_video_id, fetch_transcript, get_full_metadata, get_thumbnails,
                    get_trending_videos, search_youtube)

__version__ = "1.0.0"

__all__ = [
    "LiveSource", "LoopResult", "SYSTEM_PROMPT", "SampleSource", "SourceError", "TOOLS", "TOOL_MAPPING", "ask",
    "build_agent", "build_fixed_chain", "build_recursive_chain", "build_report_prompt", "collect_tool_calls",
    "collect_video_data", "execute_tool", "extract_video_id", "fetch_transcript", "final_answer", "get_full_metadata",
    "get_llm", "get_source", "get_thumbnails", "get_trending_videos", "run_tool_loop", "search_youtube", "set_source",
    "tool_calls", "video_report",
]
