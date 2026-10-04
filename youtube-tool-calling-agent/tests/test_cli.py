import pytest

from yt_agent.cli import main

URL = "https://www.youtube.com/watch?v=SmplAI00001"


@pytest.mark.parametrize("engine", ["agent", "manual", "recursive", "fixed"])
def test_cli_engines(capsys, engine):
    assert main([f"Summarize {URL}", "-p", "offline", "-e", engine]) == 0
    assert "Tool calling fixes that" in capsys.readouterr().out


@pytest.mark.parametrize("engine", ["agent", "manual", "recursive", "fixed"])
def test_cli_trace(capsys, engine):
    main([f"Summarize {URL}", "-p", "offline", "-e", engine, "-t"])
    out = capsys.readouterr().out
    for expected in ["Human", "extract_video_id", "fetch_transcript", "AI answer"]:
        assert expected in out


def test_cli_fixed_chain_reports_incomplete(capsys):
    main(["Show top 3 US trending videos with metadata and thumbnails", "-p", "offline", "-e", "fixed"])
    assert "fixed chain ended before an answer" in capsys.readouterr().out


def test_cli_fixed_trace_incomplete(capsys):
    main(["Show top 3 US trending videos with metadata and thumbnails", "-p", "offline", "-e", "fixed", "-t"])
    assert "(no answer)" in capsys.readouterr().out


def test_cli_report(capsys):
    assert main(["--report", "https://youtu.be/SmplCK00003", "-p", "offline"]) == 0
    out = capsys.readouterr().out
    assert "Collected: Perfect Sourdough at Home" in out and "===== VIDEO ANALYSIS =====" in out


def test_cli_report_errors(capsys):
    assert main(["--report", "https://example.com", "-p", "offline"]) == 1
    assert main(["--report", "https://youtu.be/NotThere123", "-p", "offline"]) == 0
    assert "Errors:" in capsys.readouterr().out


def test_cli_interactive(monkeypatch, capsys):
    answers = iter(["Top 2 trending in Bulgaria", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["-p", "offline"]) == 0
    out = capsys.readouterr().out
    assert "Hidden Beaches" in out and "bye" in out
