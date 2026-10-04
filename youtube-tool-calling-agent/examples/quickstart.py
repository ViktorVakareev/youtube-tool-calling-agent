"""Quickstart with LangChain's create_agent.

    python examples/quickstart.py offline              # sample videos, no key
    python examples/quickstart.py gemini live          # real model + real YouTube
"""

import sys

from dotenv import load_dotenv

from yt_agent import ask, build_agent, final_answer, get_llm, set_source, tool_calls

load_dotenv()
set_source(sys.argv[2] if len(sys.argv) > 2 else "sample")
agent = build_agent(get_llm(sys.argv[1] if len(sys.argv) > 1 else None))

for question in ["Search for videos about python testing and summarize the top result",
                 "Get top 3 youtube videos in India and their metadata"]:
    response = ask(agent, question)
    print(f"\nQ: {question}")
    for call in tool_calls(response):
        print(f"   🔧 {call['name']}({call['args']})")
    print(f"A: {final_answer(response)}")
