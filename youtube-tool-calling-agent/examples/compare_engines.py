"""Same questions, four engines - shows why the lab moves from a fixed chain to a recursive one.

    python examples/compare_engines.py offline
"""

import sys

from dotenv import load_dotenv

from yt_agent import ask, build_agent, build_fixed_chain, build_recursive_chain, get_llm, run_tool_loop

load_dotenv()
llm = get_llm(sys.argv[1] if len(sys.argv) > 1 else None)
fixed, recursive, agent = build_fixed_chain(llm), build_recursive_chain(llm), build_agent(llm)

QUESTIONS = {
    "1 round ": "Show metadata for https://youtu.be/SmplAI00002",
    "2 rounds": "Summarize https://youtu.be/SmplAI00002",
    "3 rounds": "Show top 3 US trending videos with metadata and thumbnails",
}
print(f"{'task':9} {'engine':16} {'tools run':>10}  {'LLM calls':>9}  answer")
for label, q in QUESTIONS.items():
    out = fixed.invoke({"query": q})
    runs = {
        "fixed chain": (out["messages"], out["answer"] or "— NO ANSWER (needed another round)", 3),
        "recursive chain": (msgs := recursive.invoke({"query": q}), msgs[-1].content, None),
        "manual loop": ((r := run_tool_loop(llm, q)).messages, r.answer, None),
        "create_agent": ((a := ask(agent, q)["messages"]), a[-1].content, None),
    }
    for engine, (messages, answer, llm_calls) in runs.items():
        n_llm = llm_calls or sum(1 for m in messages if m.type == "ai")
        print(f"{label:9} {engine:16} {sum(1 for m in messages if m.type == 'tool'):>10}  {n_llm:>9}  {answer.splitlines()[0][:60]}")
    print()
