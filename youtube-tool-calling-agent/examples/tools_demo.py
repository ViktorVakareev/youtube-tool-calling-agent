"""Call every tool directly - no LLM. Uses the sample catalogue unless you pass "live".

    python examples/tools_demo.py           # sample data (offline)
    python examples/tools_demo.py live      # real YouTube (pip install -e ".[youtube]")
"""

import json
import sys

from yt_agent import (extract_video_id, fetch_transcript, get_full_metadata, get_thumbnails, get_trending_videos,
                      search_youtube, set_source)

live = len(sys.argv) > 1 and sys.argv[1] == "live"
set_source("live" if live else "sample")
url = "https://www.youtube.com/watch?v=aircAruvnKk" if live else "https://www.youtube.com/watch?v=SmplAI00001"


def show(label, value):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    print(f"{label:<44} → {text[:150]}{' …' if len(text) > 150 else ''}")


video_id = extract_video_id.invoke({"url": url})
show("extract_video_id(url)", video_id)
show(f"fetch_transcript({video_id!r}, max_chars=120)", fetch_transcript.invoke({"video_id": video_id, "max_chars": 120}))
show("search_youtube('python testing', 2)", search_youtube.invoke({"query": "python testing", "max_results": 2}))
meta = get_full_metadata.invoke({"url": url})
show("get_full_metadata(url)", {k: meta.get(k) for k in ("title", "channel", "views", "duration", "likes")} if "error" not in meta else meta)
show("get_trending_videos('IN', 2)", get_trending_videos.invoke({"region_code": "IN", "max_results": 2}))
show("get_thumbnails(url) → resolutions", [t.get("resolution") for t in get_thumbnails.invoke({"url": url})])
print("\nErrors come back as data, never exceptions:")
show("fetch_transcript('NotThere123')", fetch_transcript.invoke({"video_id": "NotThere123"}))
show("get_trending_videos('FR')", get_trending_videos.invoke({"region_code": "FR"}))
