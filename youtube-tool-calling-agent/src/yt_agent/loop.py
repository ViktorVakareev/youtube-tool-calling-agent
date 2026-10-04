"""The tool-calling loop written by hand - the lab's first section.

invoke -> read AIMessage.tool_calls -> run each tool -> ToolMessage(tool_call_id) -> invoke again,
until the model answers without asking for tools (or max_steps is reached).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .tools import TOOLS

StepCallback = Callable[[str, Any], None]


def execute_tool(tool_call: dict, tools: Sequence = TOOLS) -> ToolMessage:
    """Run one tool call and wrap the result (JSON for dicts/lists) in a ToolMessage."""
    tools_by_name = {t.name: t for t in tools}
    tool = tools_by_name.get(tool_call["name"])
    try:
        if tool is None:
            raise KeyError(f"unknown tool '{tool_call['name']}'. Available: {', '.join(tools_by_name)}")
        result = tool.invoke(tool_call["args"])
        content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
    except Exception as exc:
        content = f"Error: {type(exc).__name__}: {exc}"
    return ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_call["name"])


@dataclass
class LoopResult:
    answer: str
    messages: list[BaseMessage] = field(default_factory=list)
    rounds: int = 0
    stopped: bool = False

    @property
    def tool_calls(self) -> list[dict]:
        return collect_tool_calls(self.messages)


def collect_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """[{"name", "args", "result"}, ...] for every tool call in a message list."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [{"name": c["name"], "args": c["args"], "result": results.get(c["id"])}
            for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]


def run_tool_loop(llm: BaseChatModel, query: str, tools: Sequence = TOOLS, max_steps: int = 8,
                  on_step: Optional[StepCallback] = None) -> LoopResult:
    llm_with_tools = llm.bind_tools(list(tools))
    messages: list[BaseMessage] = [HumanMessage(content=query)]
    _emit(on_step, "human", messages[0])
    for step in range(1, max_steps + 1):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            _emit(on_step, "answer", response)
            return LoopResult(text(response), messages, step - 1)
        _emit(on_step, "tool_calls", response)
        for call in response.tool_calls:
            tool_message = execute_tool(call, tools)
            messages.append(tool_message)
            _emit(on_step, "tool_result", tool_message)
    return LoopResult("Stopped: too many tool-calling rounds.", messages, max_steps, stopped=True)


def text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _emit(callback: Optional[StepCallback], kind: str, message: Any) -> None:
    if callback:
        callback(kind, message)
