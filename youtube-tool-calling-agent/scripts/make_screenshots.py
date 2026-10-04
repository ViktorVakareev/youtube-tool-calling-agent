"""Regenerate the README screenshots in docs/images/ (offline, reproducible).

    pip install -e ".[all]" ansi2html nbconvert playwright
    python scripts/make_screenshots.py

Every terminal screenshot is the *real* output of the command shown in its title bar,
run with the offline provider so no API key is needed.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "FORCE_COLOR": "1", "PY_COLORS": "1", "LLM_PROVIDER": "offline", "VIDEO_SOURCE": "sample", "COLUMNS": "120"}

TERMINAL_CSS = """
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.win{display:inline-block;min-width:860px;max-width:1180px;border-radius:10px;overflow:hidden;
     box-shadow:0 12px 32px rgba(15,23,42,.28);background:#0f172a}
.bar{background:#1e293b;padding:10px 14px;display:flex;align-items:center;gap:8px}
.dot{width:12px;height:12px;border-radius:50%}
.title{color:#94a3b8;font-size:13px;margin-left:10px;font-family:ui-monospace,Consolas,monospace}
pre{margin:0;padding:18px 22px;color:#e2e8f0;font:14px/1.55 ui-monospace,'Cascadia Code',Consolas,monospace;
    white-space:pre-wrap;word-break:break-word}
.prompt{color:#22c55e}
"""


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def terminal_html(command: str, output: str, title: str) -> str:
    conv = Ansi2HTMLConverter(inline=True, dark_bg=True)
    body = conv.convert(output, full=False)
    return f"""<html><head><meta charset="utf-8"><style>{TERMINAL_CSS}</style></head><body>
<div class="win"><div class="bar"><span class="dot" style="background:#ef4444"></span>
<span class="dot" style="background:#f59e0b"></span><span class="dot" style="background:#22c55e"></span>
<span class="title">{html.escape(title)}</span></div>
<pre><span class="prompt">$</span> {html.escape(command)}\n{body}</pre></div></body></html>"""


def shoot(page, html_doc: str, name: str, selector: str = ".win") -> None:
    page.set_content(html_doc, wait_until="networkidle")
    page.locator(selector).first.screenshot(path=str(OUT / name))
    print("  ✓", name)


ARCHITECTURE = """<html><head><meta charset="utf-8"><style>
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.card{display:inline-block;background:#fff;border-radius:12px;padding:22px 26px;box-shadow:0 12px 32px rgba(15,23,42,.18)}
h2{margin:0 0 4px;font-size:20px;color:#0f172a}p{margin:0 0 14px;color:#475569;font-size:14px}
text{font-family:-apple-system,Segoe UI,sans-serif} .mono text, text.mono{font-family:ui-monospace,Consolas,monospace}
</style></head><body><div class="card">
<h2>Four ways to drive the same YouTube tools</h2>
<p>The engines differ only in <b>who decides the next tool call</b> and <b>how many rounds</b> they allow. The tools and the data source are shared.</p>
<svg width="1100" height="470" viewBox="0 0 1100 470" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<rect x="10" y="150" width="170" height="90" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
<text x="95" y="182" text-anchor="middle" font-size="15" font-weight="700" fill="#075985">Question</text>
<text x="95" y="206" text-anchor="middle" font-size="12.5" fill="#0c4a6e">"Top 3 trending in India</text>
<text x="95" y="224" text-anchor="middle" font-size="12.5" fill="#0c4a6e">with metadata"</text>
<g font-size="13">
<rect x="240" y="20" width="300" height="62" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<text x="256" y="45" font-weight="700" fill="#92400e">Manual loop</text><text x="256" y="68" fill="#78350f">invoke → tool_calls → ToolMessage → repeat</text>
<rect x="240" y="96" width="300" height="62" rx="10" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="256" y="121" font-weight="700" fill="#991b1b">Fixed LCEL chain</text><text x="256" y="144" fill="#7f1d1d">exactly 2 tool rounds - breaks on 1 or 3</text>
<rect x="240" y="172" width="300" height="62" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="256" y="197" font-weight="700" fill="#166534">Recursive chain</text><text x="256" y="220" fill="#14532d">repeat until no tool calls (max_rounds guard)</text>
<rect x="240" y="248" width="300" height="62" rx="10" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>
<text x="256" y="273" font-weight="700" fill="#5b21b6">create_agent (LangChain 1.x)</text><text x="256" y="296" fill="#4c1d95">the same loop, prebuilt</text>
</g>
<text x="390" y="336" text-anchor="middle" font-size="12" fill="#475569">LLM: Gemini · OpenAI · offline stand-in</text>
<rect x="640" y="20" width="250" height="290" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="765" y="50" text-anchor="middle" font-size="15" font-weight="700" fill="#6b21a8">6 tools (@tool)</text>
<g class="mono" font-size="12.5" fill="#581c87">
<text x="660" y="84">extract_video_id</text><text x="660" y="116">fetch_transcript</text><text x="660" y="148">search_youtube</text>
<text x="660" y="180">get_full_metadata</text><text x="660" y="212">get_trending_videos</text><text x="660" y="244">get_thumbnails</text></g>
<text x="660" y="282" font-size="11.5" fill="#7e22ce">errors returned as data</text>
<text x="660" y="298" font-size="11.5" fill="#7e22ce">transcripts capped (max_chars)</text>
<rect x="950" y="40" width="140" height="110" rx="12" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
<text x="1020" y="70" text-anchor="middle" font-size="14" font-weight="700" fill="#991b1b">LiveSource</text>
<text x="1020" y="94" text-anchor="middle" font-size="12" fill="#7f1d1d">yt-dlp +</text>
<text x="1020" y="112" text-anchor="middle" font-size="12" fill="#7f1d1d">youtube-transcript-api</text>
<text x="1020" y="134" text-anchor="middle" font-size="11" fill="#7f1d1d">real YouTube</text>
<rect x="950" y="180" width="140" height="110" rx="12" fill="#e2e8f0" stroke="#475569" stroke-width="2"/>
<text x="1020" y="210" text-anchor="middle" font-size="14" font-weight="700" fill="#1e293b">SampleSource</text>
<text x="1020" y="234" text-anchor="middle" font-size="12" fill="#334155">6 made-up videos</text>
<text x="1020" y="252" text-anchor="middle" font-size="12" fill="#334155">bundled JSON</text>
<text x="1020" y="274" text-anchor="middle" font-size="11" fill="#334155">offline · tests · demos</text>
<line x1="180" y1="195" x2="236" y2="195" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<line x1="540" y1="165" x2="636" y2="165" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="548" y="155" font-size="12" fill="#334155">tool_calls</text>
<line x1="890" y1="110" x2="946" y2="96" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<line x1="890" y1="210" x2="946" y2="232" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<rect x="240" y="370" width="850" height="80" rx="12" fill="#fff7ed" stroke="#ea580c" stroke-width="2" stroke-dasharray="6 4"/>
<text x="260" y="398" font-size="14" font-weight="700" fill="#9a3412">Exercises pipeline (no tool calling)</text>
<text x="260" y="424" font-size="12.5" fill="#7c2d12">your code runs extract_video_id → get_full_metadata → fetch_transcript → get_thumbnails,</text>
<text x="260" y="442" font-size="12.5" fill="#7c2d12">builds one prompt, and makes a single llm.invoke() for the analysis - predictable and cheap when the steps are known.</text>
</svg></div></body></html>"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        shoot(page, ARCHITECTURE, "architecture.png", ".card")

        py = sys.executable
        shots = [
            ("tools-direct.png", "Tools called directly (sample catalogue, no LLM)", f"{py} examples/tools_demo.py", "python examples/tools_demo.py"),
            ("manual-step-by-step.png", "The lab's manual loop: summarize a video step by step",
             f"{py} examples/manual_loop_step_by_step.py offline", "python examples/manual_loop_step_by_step.py offline"),
            ("trace-recursive-trending.png", "Recursive chain: 3 tool rounds, 7 tool calls",
             'yt-agent "Show top 3 US trending videos with metadata and thumbnails" -p offline -e recursive -t', None),
            ("trace-agent-search.png", "create_agent: search, then summarize; and an error handled",
             'yt-agent "Search for videos about python testing and summarize the top result" -p offline -t && yt-agent "Summarize https://youtu.be/NotThere123" -p offline -t', None),
            ("compare-engines.png", "Same questions, four engines",
             f"{py} examples/compare_engines.py offline", "python examples/compare_engines.py offline"),
            ("video-report.png", "Exercises 1-7: collect the data, then one LLM call",
             f"{py} examples/video_report_exercises.py offline", "python examples/video_report_exercises.py offline"),
            ("pytest-results.png", "Test suite", f"{py} -m pytest -v --color=yes -p no:cacheprovider", "pytest -v"),
            ("pytest-coverage.png", "Coverage",
             f"{py} -m pytest -q --color=yes -p no:cacheprovider --cov=yt_agent --cov-report=term-missing",
             "pytest --cov=yt_agent --cov-report=term-missing"),
        ]
        for name, title, cmd, shown in shots:
            shoot(page, terminal_html(shown or cmd, run(cmd), title), name)

        nb_html = run(f"{py} -m jupyter nbconvert --to html --stdout notebooks/walkthrough.ipynb 2>/dev/null")
        page.set_viewport_size({"width": 1100, "height": 900})
        page.set_content(nb_html, wait_until="networkidle")
        page.screenshot(path=str(OUT / "notebook-walkthrough.png"), full_page=True)
        print("  ✓ notebook-walkthrough.png")
        browser.close()


if __name__ == "__main__":
    main()
