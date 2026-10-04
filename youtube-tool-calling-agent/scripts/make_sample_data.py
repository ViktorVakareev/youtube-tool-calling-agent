"""Generate data/sample_videos.json - a small, made-up video catalogue (no real YouTube content).

    python scripts/make_sample_data.py
"""

import json
from pathlib import Path

VIDEOS = [
    {
        "video_id": "SmplAI00001", "title": "Tool Calling in 10 Minutes", "channel": "Sample Academy",
        "views": 184200, "duration": 612, "likes": 9100, "comments": 412, "published": "2026-03-02",
        "tags": ["ai", "llm", "tool calling", "langchain", "agents"],
        "description": "How large language models call your Python functions.",
        "chapters": [{"title": "Why tools", "start_time": 0}, {"title": "The tool call", "start_time": 150},
                     {"title": "The loop", "start_time": 380}],
        "transcripts": {
            "en": "A language model only produces text, so on its own it cannot check a database or do exact math. "
                  "Tool calling fixes that by letting the model ask for a function by name. "
                  "Your code runs the function and sends the result back with the same call id. "
                  "The model then reads the result and either asks for another tool or writes the final answer. "
                  "Repeating this until no more tools are needed is called the agent loop. "
                  "Good tool names and docstrings matter, because they are all the model sees.",
            "es": "Un modelo de lenguaje solo produce texto. Las herramientas le permiten pedir una funcion por su nombre. "
                  "Tu codigo ejecuta la funcion y devuelve el resultado con el mismo identificador.",
        },
    },
    {
        "video_id": "SmplAI00002", "title": "What Is an AI Agent?", "channel": "Sample Academy",
        "views": 932500, "duration": 845, "likes": 41800, "comments": 2210, "published": "2026-01-15",
        "tags": ["ai", "agents", "llm", "automation"],
        "description": "Agents explained without the hype.",
        "chapters": [{"title": "Definition", "start_time": 0}, {"title": "Examples", "start_time": 300}],
        "transcripts": {
            "en": "An AI agent is a language model that can take actions, not just answer questions. "
                  "It plans a step, calls a tool, looks at the result and decides what to do next. "
                  "Agents are useful when a task needs fresh data or several steps. "
                  "They are a poor fit when a single prompt already gives a reliable answer. "
                  "Every extra step adds cost and another chance to go wrong, so keep agents focused.",
        },
    },
    {
        "video_id": "SmplCK00003", "title": "Perfect Sourdough at Home", "channel": "Kitchen Lab",
        "views": 2410000, "duration": 1320, "likes": 88000, "comments": 5400, "published": "2025-11-20",
        "tags": ["cooking", "bread", "sourdough", "baking"],
        "description": "A beginner-friendly sourdough method.",
        "chapters": [{"title": "Starter", "start_time": 0}, {"title": "Shaping", "start_time": 600},
                     {"title": "Baking", "start_time": 1000}],
        "transcripts": {
            "en": "Sourdough needs only flour, water, salt and an active starter. "
                  "Feed the starter the night before so it is bubbly when you mix the dough. "
                  "Stretch and fold the dough every half hour to build strength without kneading. "
                  "A long cold proof in the fridge gives better flavour and an easier bake. "
                  "Bake in a very hot covered pot first, then uncover it to get a dark crust.",
        },
    },
    {
        "video_id": "SmplFT00004", "title": "20-Minute Morning Mobility", "channel": "Move Daily",
        "views": 655000, "duration": 1205, "likes": 30200, "comments": 980, "published": "2026-02-08",
        "tags": ["fitness", "mobility", "stretching", "morning routine"],
        "description": "A gentle full-body routine to start the day.",
        "chapters": [{"title": "Neck and shoulders", "start_time": 0}, {"title": "Hips", "start_time": 420}],
        "transcripts": {
            "en": "Start slowly with neck circles and shoulder rolls to wake up the upper body. "
                  "Move into cat cow stretches to loosen the spine. "
                  "Hip openers help if you sit for long hours at a desk. "
                  "Breathe steadily and never force a stretch into pain. "
                  "Ten minutes every morning beats one long session a week.",
        },
    },
    {
        "video_id": "SmplTR00005", "title": "Hidden Beaches of the Black Sea", "channel": "Wander Notes",
        "views": 318000, "duration": 980, "likes": 15600, "comments": 640, "published": "2025-08-30",
        "tags": ["travel", "beaches", "black sea", "bulgaria", "summer"],
        "description": "Quiet coves away from the crowds.",
        "chapters": [{"title": "North coast", "start_time": 0}, {"title": "South coast", "start_time": 500}],
        "transcripts": {
            "en": "The Black Sea coast has many quiet beaches away from the big resorts. "
                  "In the north, small coves sit below limestone cliffs. "
                  "The south has wide sandy bays backed by forest and river mouths. "
                  "Go in June or September for warm water and fewer people. "
                  "Bring cash, because small beach bars often do not take cards.",
        },
    },
    {
        "video_id": "SmplPY00006", "title": "Python Testing with pytest", "channel": "Sample Academy",
        "views": 271300, "duration": 1500, "likes": 14900, "comments": 530, "published": "2026-04-11",
        "tags": ["python", "testing", "pytest", "programming"],
        "description": "From your first test to fixtures and parametrize.",
        "chapters": [{"title": "First test", "start_time": 0}, {"title": "Fixtures", "start_time": 520},
                     {"title": "Parametrize", "start_time": 1040}],
        "transcripts": {
            "en": "A pytest test is just a function whose name starts with test. "
                  "Plain assert statements give clear failure messages without extra helpers. "
                  "Fixtures provide shared setup such as temporary files or fake clients. "
                  "Parametrize runs one test against many inputs, which keeps edge cases visible. "
                  "Fast tests that run on every commit are worth more than slow ones nobody runs.",
        },
    },
]

for v in VIDEOS:
    v["thumbnails"] = [
        {"url": f"https://i.ytimg.com/vi/{v['video_id']}/{name}.jpg", "width": w, "height": h}
        for name, w, h in [("default", 120, 90), ("mqdefault", 320, 180), ("hqdefault", 480, 360), ("maxresdefault", 1280, 720)]
    ]

TRENDING = {
    "US": ["SmplAI00001", "SmplFT00004", "SmplCK00003", "SmplPY00006"],
    "IN": ["SmplAI00002", "SmplPY00006", "SmplAI00001", "SmplCK00003"],
    "GB": ["SmplTR00005", "SmplCK00003", "SmplAI00002", "SmplFT00004"],
    "BG": ["SmplTR00005", "SmplAI00002", "SmplFT00004", "SmplCK00003"],
}

out = Path(__file__).resolve().parents[1] / "src" / "yt_agent" / "data" / "sample_videos.json"
out.write_text(json.dumps({"note": "Made-up sample catalogue for offline demos and tests. Not real YouTube data.",
                           "videos": VIDEOS, "trending": TRENDING}, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(VIDEOS), "videos")
