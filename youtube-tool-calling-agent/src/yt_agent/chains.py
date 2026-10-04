"""The lab's LCEL chains: a fixed two-round chain and a recursive (universal) chain.

Fixed chain - exactly two tool rounds wired with RunnablePassthrough.assign:
    query -> LLM -> tools -> LLM -> tools -> LLM -> answer
Great for "extract ID -> fetch transcript -> summarize", wrong for anything needing
one round or three.

Recursive chain - repeats "run tools -> ask LLM" until the model stops calling tools,
so it handles any number of rounds. Added here: a ``max_rounds`` guard, because the
lab's version could recurse forever (or hit Python's recursion limit).
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from .loop import execute_tool, text
from .tools import TOOLS


def build_fixed_chain(llm: BaseChatModel, tools: Sequence = TOOLS) -> Runnable:
    """{"query": str} -> {"messages": [...], "answer": str}. Always exactly two tool rounds."""
    llm_with_tools = llm.bind_tools(list(tools))

    def run_tools(ai: AIMessage) -> list:
        return [execute_tool(tc, tools) for tc in ai.tool_calls]

    return (
        RunnablePassthrough.assign(messages=lambda x: [HumanMessage(content=x["query"])])
        | RunnablePassthrough.assign(ai_response=lambda x: llm_with_tools.invoke(x["messages"]))
        | RunnablePassthrough.assign(tool_messages=lambda x: run_tools(x["ai_response"]))
        | RunnablePassthrough.assign(messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"])
        | RunnablePassthrough.assign(ai_response2=lambda x: llm_with_tools.invoke(x["messages"]))
        | RunnablePassthrough.assign(tool_messages2=lambda x: run_tools(x["ai_response2"]))
        | RunnablePassthrough.assign(messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"])
        | RunnablePassthrough.assign(final=lambda x: llm_with_tools.invoke(x["messages"]))
        | RunnableLambda(lambda x: {"messages": x["messages"] + [x["final"]], "answer": text(x["final"]),
                                    "complete": not x["final"].tool_calls})
    )


def build_recursive_chain(llm: BaseChatModel, tools: Sequence = TOOLS, max_rounds: int = 8) -> Runnable:
    """{"query": str} -> list of messages; the last one is the final answer."""
    llm_with_tools = llm.bind_tools(list(tools))

    def should_continue(messages: list[BaseMessage]) -> bool:
        return bool(getattr(messages[-1], "tool_calls", None))

    def process_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
        tool_messages = [execute_tool(tc, tools) for tc in messages[-1].tool_calls]
        updated = messages + tool_messages
        return updated + [llm_with_tools.invoke(updated)]

    def recurse(messages: list[BaseMessage], rounds: int = 0) -> list[BaseMessage]:
        if not should_continue(messages):
            return messages
        if rounds >= max_rounds:
            return messages + [AIMessage(content=f"Stopped after {max_rounds} tool rounds.")]
        return recurse(process_tool_calls(messages), rounds + 1)

    return (
        RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
        | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
        | RunnableLambda(recurse)
    )
