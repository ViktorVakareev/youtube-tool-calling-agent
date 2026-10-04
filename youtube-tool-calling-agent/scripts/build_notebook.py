"""Generate and execute notebooks/walkthrough.ipynb (offline model + sample videos by default).

    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# YouTube tool-calling agent — walkthrough\n\n"
       "Tools → the manual loop → a fixed LCEL chain → a recursive chain → `create_agent` → the exercises' report. "
       "Runs **offline by default**: a rule-based stand-in model and a bundled catalogue of made-up videos (no API key, no network, no cost). "
       "Set `PROVIDER = \"gemini\"` / `\"openai\"` and `SOURCE = \"live\"` for the real thing."),
    code("PROVIDER, SOURCE = \"offline\", \"sample\"   # \"gemini\" | \"openai\" | \"offline\";  \"live\" | \"sample\"\n\n"
         "from dotenv import load_dotenv\nload_dotenv()\n"
         "from langchain_core.messages import HumanMessage\n"
         "from yt_agent import (TOOLS, extract_video_id, fetch_transcript, get_full_metadata, get_llm, set_source,\n"
         "                      execute_tool, run_tool_loop, build_fixed_chain, build_recursive_chain, build_agent, ask,\n"
         "                      collect_tool_calls, collect_video_data, build_report_prompt)\n"
         "from yt_agent.cli import print_step, print_trace\n"
         "set_source(SOURCE)\nllm = get_llm(PROVIDER)\nURL = \"https://www.youtube.com/watch?v=SmplAI00001\""),
    md("## 1. The tools (what the model sees)"),
    code("for t in TOOLS:\n    print(f\"{t.name:20} {list(t.args)}\")"),
    code("vid = extract_video_id.invoke({\"url\": URL})\nprint(vid)\n"
         "print(fetch_transcript.invoke({\"video_id\": vid, \"max_chars\": 120}))\n"
         "print(get_full_metadata.invoke({\"url\": \"https://youtu.be/NotThere123\"}))   # errors are data"),
    md("## 2. The manual loop: the model asks, your code runs the tool, the result goes back with the call id"),
    code("result = run_tool_loop(llm, f\"I want to summarize youtube video: {URL} in english\", on_step=print_step)"),
    md("## 3. Fixed chain (two tool rounds, wired with `RunnablePassthrough.assign`)\n"
       "Fine for *ID → transcript → summary*; stuck when a task needs a third round."),
    code("fixed = build_fixed_chain(llm)\nfor q in [f\"Summarize {URL}\", \"Show top 3 US trending videos with metadata and thumbnails\"]:\n"
         "    out = fixed.invoke({\"query\": q})\n    print(f\"complete={out['complete']!s:5}  answer={out['answer'][:70]!r}\")"),
    md("## 4. Recursive chain: keep going until the model stops calling tools"),
    code("messages = build_recursive_chain(llm).invoke({\"query\": \"Show top 3 US trending videos with metadata and thumbnails\"})\n"
         "print_trace(messages)"),
    md("## 5. The same loop, prebuilt: `create_agent`"),
    code("print_trace(ask(build_agent(llm), \"Search for videos about python testing and summarize the top result\")[\"messages\"])"),
    md("## 6. Exercises: collect everything, then one LLM call (no tool calling)"),
    code("data = collect_video_data(\"https://www.youtube.com/watch?v=SmplCK00003\")\n"
         "print(data[\"video_id\"], \"|\", data[\"metadata\"][\"title\"], \"|\", len(data[\"transcript\"]), \"chars |\", len(data[\"thumbnails\"]), \"thumbnails\")\n"
         "response = llm.invoke([HumanMessage(content=build_report_prompt(data))])\n"
         "print(\"\\n===== VIDEO ANALYSIS =====\\n\")\nprint(response.content)"),
    md("## 7. Evaluate: check the tool calls, not the wording"),
    code("checks = [\n"
         "    (f\"Summarize {URL}\", [\"extract_video_id\", \"fetch_transcript\"]),\n"
         "    (\"Get top 3 youtube videos in India and their metadata\", [\"get_trending_videos\"] + [\"get_full_metadata\"] * 3),\n"
         "]\npassed = 0\nfor q, expected in checks:\n    names = [c[\"name\"] for c in run_tool_loop(llm, q).tool_calls]\n"
         "    ok = names == expected\n    passed += ok\n    print(\"✅\" if ok else \"❌\", q[:55], \"->\", names)\n"
         "print(f\"\\n{passed}/{len(checks)} checks passed\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, ROOT / "notebooks" / "walkthrough.ipynb")
print("wrote notebooks/walkthrough.ipynb")
