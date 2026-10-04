"""The lab's exercises 1-7: collect everything about one video, then ONE LLM call.

    python examples/video_report_exercises.py offline
    python examples/video_report_exercises.py gemini live https://www.youtube.com/watch?v=aircAruvnKk
"""

import sys

from dotenv import load_dotenv

from yt_agent import build_report_prompt, collect_video_data, get_llm, set_source
from yt_agent.loop import text
from langchain_core.messages import HumanMessage

load_dotenv()
provider = sys.argv[1] if len(sys.argv) > 1 else None
set_source(sys.argv[2] if len(sys.argv) > 2 else "sample")
youtube_url = sys.argv[3] if len(sys.argv) > 3 else "https://www.youtube.com/watch?v=SmplTR00005"   # Exercise 1

data = collect_video_data(youtube_url)                                   # Exercises 2-5
print(f"Ex 2  video ID     : {data.get('video_id')}")
m = data.get("metadata", {})
print(f"Ex 3  metadata     : {m.get('title')} | {m.get('channel')} | {m.get('views')} views | {m.get('duration')} s")
print(f"Ex 4  transcript   : {len(data.get('transcript', ''))} characters")
print(f"Ex 5  thumbnails   : {len(data.get('thumbnails', []))}")
if data.get("errors"):
    print("      errors       :", data["errors"])

prompt = build_report_prompt(data)                                        # comprehensive prompt
response = get_llm(provider).invoke([HumanMessage(content=prompt)])       # Exercise 6: one call, no tools
print("\n===== VIDEO ANALYSIS =====\n")                                   # Exercise 7
print(text(response))
