import pytest

from yt_agent.sources import (TRENDING_RETIRED, LiveSource, SampleSource, SourceError, extract_video_id, get_source,
                              make_source, set_source)


@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/watch?v=hfIUstzHs9A", "hfIUstzHs9A"),
    ("https://youtu.be/qWHaMrR5WHQ", "qWHaMrR5WHQ"),
    ("https://www.youtube.com/embed/T-D1OfcDW1M?start=3", "T-D1OfcDW1M"),
    ("https://www.youtube.com/shorts/abc_DEF-123", "abc_DEF-123"),
    ("https://www.youtube.com/watch?feature=share&v=t97ipSIDEfU&t=10", "t97ipSIDEfU"),
    ("SmplAI00001", "SmplAI00001"),
    ("https://example.com/video", None), ("not a url", None), ("", None), (None, None),
])
def test_extract_video_id(url, expected):
    assert extract_video_id(url) == expected


class TestSampleSource:
    def test_search_ranks_by_relevance(self):
        hits = SampleSource().search("python testing")
        assert hits[0]["video_id"] == "SmplPY00006" and set(hits[0]) == {"title", "video_id", "url", "channel", "views", "duration"}
        assert SampleSource().search("quantum knitting") == []

    def test_transcript_languages(self):
        s = SampleSource()
        assert s.transcript("SmplAI00001").startswith("A language model")
        assert s.transcript("SmplAI00001", "es").startswith("Un modelo")
        with pytest.raises(SourceError, match="Available: en"):
            s.transcript("SmplCK00003", "es")

    def test_metadata_trending_thumbnails(self):
        s = SampleSource()
        meta = s.metadata("SmplCK00003")
        assert meta["title"] == "Perfect Sourdough at Home" and meta["duration"] == 1320 and len(meta["chapters"]) == 3
        assert [v["video_id"] for v in s.trending("in", 2)] == ["SmplAI00002", "SmplPY00006"]
        assert s.thumbnails("SmplCK00003")[-1]["resolution"] == "1280x720"

    def test_unknown_video_and_region(self):
        with pytest.raises(SourceError, match="not found"):
            SampleSource().metadata("NotThere123")
        with pytest.raises(SourceError, match="Available: US, IN, GB, BG"):
            SampleSource().trending("FR")


class TestLiveSource:
    def test_search_metadata_thumbnails_transcript(self, fake_youtube_libs):
        live = LiveSource()
        assert live.search("x") == [{"title": "Found", "video_id": "abcdefghijk", "url": "https://www.youtube.com/watch?v=abcdefghijk",
                                     "channel": "Ch", "views": 9, "duration": 30}]
        meta = live.metadata("abcdefghijk")
        assert meta["title"] == "Live title" and meta["chapters"] == [] and meta["likes"] == 7
        assert [t["resolution"] for t in live.thumbnails("abcdefghijk")] == ["120x90", None]
        assert live.transcript("abcdefghijk") == "hello world"

    def test_trending_works_while_feed_returns_entries(self, fake_youtube_libs):
        assert LiveSource().trending("us")[0]["video_id"] == "abcdefghijk"

    def test_trending_reports_retired_page(self, fake_youtube_libs):
        fake_youtube_libs["trending"] = []
        with pytest.raises(SourceError, match="retired its Trending page"):
            LiveSource().trending("US")
        fake_youtube_libs["fail"] = True
        with pytest.raises(SourceError) as err:
            LiveSource().trending("US")
        assert str(err.value) == TRENDING_RETIRED

    def test_failures_become_source_errors(self, fake_youtube_libs):
        fake_youtube_libs["fail"] = True
        fake_youtube_libs["transcript_fail"] = True
        live = LiveSource()
        for call in (lambda: live.search("x"), lambda: live.metadata("abcdefghijk"), lambda: live.transcript("abcdefghijk")):
            with pytest.raises(SourceError):
                call()


def test_source_switching(monkeypatch, fake_youtube_libs):
    assert isinstance(set_source("live"), LiveSource) and get_source().name == "live"
    set_source("sample")
    assert get_source().name == "sample"
    with pytest.raises(ValueError):
        make_source("tiktok")
    import yt_agent.sources as sources
    monkeypatch.setattr(sources, "_SOURCE", None)
    monkeypatch.setenv("VIDEO_SOURCE", "sample")
    assert get_source().name == "sample"
