import sys
import types

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from yt_agent.sources import SampleSource, set_source


@pytest.fixture(autouse=True)
def sample_source(monkeypatch):
    """Every test starts on the bundled sample catalogue, even if a local .env says VIDEO_SOURCE=live."""
    monkeypatch.setenv("VIDEO_SOURCE", "sample")
    source = set_source(SampleSource())
    yield source
    set_source(SampleSource())


class ScriptedToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def calls(*specs):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{n}_{i}"} for i, (n, a) in enumerate(specs)])


@pytest.fixture
def scripted_model():
    return lambda *messages: ScriptedToolModel(messages=iter(messages))


@pytest.fixture
def fake_youtube_libs(monkeypatch):
    """Stub yt_dlp + youtube_transcript_api so LiveSource can be tested without network."""
    state = {"trending": [{"id": "abcdefghijk", "title": "T1", "uploader": "C", "view_count": 5, "duration": 60}],
             "fail": False, "transcript_fail": False}

    class YoutubeDL:
        def __init__(self, opts): self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False):
            if state["fail"]:
                raise RuntimeError("network down")
            if url.startswith("ytsearch"):
                return {"entries": [{"id": "abcdefghijk", "title": "Found", "channel": "Ch", "view_count": 9, "duration": 30},
                                    {"title": "no id"}]}
            if "feed/trending" in url:
                return {"entries": state["trending"]}
            return {"title": "Live title", "uploader": "Live channel", "view_count": 100, "duration": 61,
                    "like_count": 7, "comment_count": 2, "upload_date": "20260101", "chapters": None,
                    "thumbnails": [{"url": "https://x/1.jpg", "width": 120, "height": 90}, {"id": "nourl"},
                                   {"url": "https://x/2.jpg"}]}

    class YouTubeTranscriptApi:
        def fetch(self, video_id, languages=("en",)):
            if state["transcript_fail"]:
                raise LookupError("disabled")
            return types.SimpleNamespace(snippets=[types.SimpleNamespace(text="hello"), types.SimpleNamespace(text="world")])

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=YoutubeDL))
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", types.SimpleNamespace(YouTubeTranscriptApi=YouTubeTranscriptApi))
    return state
