"""Provider switch and the LangChain 1.x agent (create_agent runs the tool loop for you)."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a YouTube research assistant. Use the tools to get real data - never invent titles, numbers "
    "or video content. To summarize a video: extract the video ID, fetch the transcript, then summarize it. "
    "For lists of videos, get metadata or thumbnails only for the videos the user asked about. "
    "If a tool returns an error, say so plainly. Keep answers concise and cite video titles."
)

PROVIDERS = ("gemini", "openai", "offline")
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
    """Build the chat model. ``provider`` defaults to $LLM_PROVIDER, then "gemini".

    - gemini:  free API key from Google AI Studio in $GOOGLE_API_KEY ($GEMINI_MODEL)
    - openai:  $OPENAI_API_KEY ($OPENAI_MODEL), billed per token
    - offline: rule-based stand-in, no key, no network, no cost
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODELS["gemini"]),
                                      temperature=0, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"]), temperature=0, **kwargs)
    if provider == "offline":
        from .offline import OfflineYouTubeModel

        return OfflineYouTubeModel()
    raise ValueError(f"Unknown provider {provider!r}. Choose one of {PROVIDERS}.")


def build_agent(llm: Optional[BaseChatModel] = None, tools: Optional[Sequence] = None,
                system_prompt: str = SYSTEM_PROMPT):
    """LangChain's prebuilt loop: model -> tools -> model ... until a final answer."""
    return create_agent(model=llm or get_llm(), tools=list(tools or TOOLS), system_prompt=system_prompt)


def ask(agent, question: str, history: Optional[list] = None) -> dict:
    return agent.invoke({"messages": [*(history or []), ("human", question)]})


def final_answer(response: dict) -> str:
    content = response["messages"][-1].content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def tool_calls(response: dict) -> list[dict]:
    """Tool calls of the last turn: [{"name", "args", "result"}, ...]."""
    messages = response["messages"]
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    messages = messages[last_human:]
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [{"name": c["name"], "args": c["args"], "result": results.get(c["id"])}
            for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
