"""Command line for the YouTube tool-calling agent.

    yt-agent "Summarize https://www.youtube.com/watch?v=SmplAI00001" -p offline -t
    yt-agent "Top 3 trending videos in India with metadata" -p offline -e recursive -t
    yt-agent --report https://www.youtube.com/watch?v=SmplCK00003 -p offline
    yt-agent -p gemini -s live                      # interactive, real YouTube ('exit' to quit)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .agent import PROVIDERS, ask, build_agent, get_llm
from .chains import build_fixed_chain, build_recursive_chain
from .loop import run_tool_loop, text
from .report import video_report
from .sources import set_source

ENGINES = ("agent", "manual", "recursive", "fixed")
_COLOR = sys.stdout.isatty() or bool(os.getenv("FORCE_COLOR"))


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def _short(s: str, limit: int = 180) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + f" … (+{len(s) - limit} chars)"


def _label(code: str, name: str) -> str:
    return _c(code, name.ljust(12))


def print_step(kind: str, msg) -> None:
    if kind == "human":
        print(_label("1;36", "Human"), msg.content)
    elif kind == "tool_calls":
        label = "AI → tool" if len(msg.tool_calls) == 1 else f"AI → {len(msg.tool_calls)} tools"
        for call in msg.tool_calls:
            print(_label("1;33", label), _c("33", call["name"]) + f"({call['args']})")
    elif kind == "tool_result":
        print(_label("1;35", "Tool"), _c("35", msg.name or ""), "→", _short(msg.content))
    elif kind == "answer":
        print(_label("1;32", "AI answer"), textwrap.indent(text(msg), " " * 13).lstrip())


def print_trace(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            print_step("human", msg)
        elif isinstance(msg, AIMessage):
            print_step("tool_calls" if msg.tool_calls else "answer", msg)
        elif isinstance(msg, ToolMessage):
            print_step("tool_result", msg)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="yt-agent", description="YouTube tool-calling agent (LangChain)")
    parser.add_argument("question", nargs="*", help="question; omit for interactive chat")
    parser.add_argument("-p", "--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "gemini"))
    parser.add_argument("-m", "--model", default=None, help="override the model name")
    parser.add_argument("-s", "--source", choices=("sample", "live"), default=os.getenv("VIDEO_SOURCE", "sample"),
                        help="sample = bundled made-up videos (offline); live = real YouTube")
    parser.add_argument("-e", "--engine", choices=ENGINES, default="agent",
                        help="agent = create_agent; manual = hand-written loop; recursive / fixed = the lab's LCEL chains")
    parser.add_argument("-t", "--trace", action="store_true", help="show every message")
    parser.add_argument("-r", "--report", metavar="URL", help="run the exercises pipeline: collect data, then one LLM call")
    args = parser.parse_args(argv)

    set_source(args.source)
    llm = get_llm(args.provider, args.model)

    if args.report:
        result = video_report(args.report, llm)
        if result.get("analysis") is None:
            print(_c("1;31", "Error: ") + result["error"])
            return 1
        m = result["metadata"]
        print(_c("1;36", "Collected:"), f"{m.get('title')} | {m.get('channel')} | {m.get('views')} views | "
              f"transcript {len(result['transcript'])} chars | {len(result['thumbnails'])} thumbnails")
        if result["errors"]:
            print(_c("1;31", "Errors:"), "; ".join(result["errors"]))
        print(_c("1;32", "\n===== VIDEO ANALYSIS =====\n") + result["analysis"])
        return 0

    agent = build_agent(llm) if args.engine == "agent" else None
    fixed = build_fixed_chain(llm) if args.engine == "fixed" else None
    recursive = build_recursive_chain(llm) if args.engine == "recursive" else None

    def run(question: str) -> None:
        if args.engine == "manual":
            result = run_tool_loop(llm, question, on_step=print_step if args.trace else None)
            messages, answer = result.messages, result.answer
        elif args.engine == "fixed":
            out = fixed.invoke({"query": question})
            messages, answer = out["messages"], out["answer"] or "(fixed chain ended before an answer - the task needed a different number of tool rounds)"
        elif args.engine == "recursive":
            messages = recursive.invoke({"query": question})
            answer = text(messages[-1])
        else:
            messages = ask(agent, question)["messages"]
            answer = text(messages[-1])
        if args.trace and args.engine != "manual":
            print_trace(messages)
            if args.engine == "fixed" and not messages[-1].content:
                print(_c("1;31", "(no answer)") + " " + answer)
        elif not args.trace:
            print(_c("1;32", "Agent: ") + answer)

    if args.question:
        run(" ".join(args.question))
        return 0

    print(f"📺 YouTube agent {_c('2', f'(provider: {args.provider}, source: {args.source}, engine: {args.engine})')} - 'exit' to quit")
    while True:
        try:
            question = input(_c("1;36", "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        run(question)
    print("bye")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
