from yt_agent import build_report_prompt, collect_video_data, video_report
from yt_agent.offline import OfflineYouTubeModel


def test_collect_video_data():
    data = collect_video_data("https://youtu.be/SmplFT00004")
    assert data["video_id"] == "SmplFT00004" and data["metadata"]["channel"] == "Move Daily"
    assert data["transcript"].startswith("Start slowly") and len(data["thumbnails"]) == 4 and data["errors"] == []


def test_collect_records_errors_instead_of_raising():
    assert collect_video_data("https://example.com")["error"] == "Error: Invalid YouTube URL"
    data = collect_video_data("https://youtu.be/NotThere123")
    assert len(data["errors"]) == 3 and data["transcript"] == "" and data["thumbnails"] == []


def test_prompt_contains_everything():
    prompt = build_report_prompt(collect_video_data("https://youtu.be/SmplCK00003"))
    for expected in ["VIDEO TITLE: Perfect Sourdough at Home", "CHANNEL: Kitchen Lab", "VIEWS: 2410000",
                     "DURATION: 22 min 0 s", "LIKES: 88000", "THUMBNAILS AVAILABLE: 4", "TRANSCRIPT EXCERPT:",
                     "Sourdough needs only flour", "4. A brief analysis"]:
        assert expected in prompt


def test_prompt_handles_missing_data():
    prompt = build_report_prompt({"metadata": {}, "transcript": "", "thumbnails": []})
    assert "VIDEO TITLE: unknown" in prompt and "(no transcript available)" in prompt and "DURATION: unknown" in prompt


def test_video_report_single_llm_call():
    report = video_report("https://youtu.be/SmplCK00003", OfflineYouTubeModel())
    assert report["analysis"].startswith('Analysis of "Perfect Sourdough at Home"')
    assert "3.7% like-to-view ratio" in report["analysis"]
    assert video_report("bad", OfflineYouTubeModel())["analysis"] is None
