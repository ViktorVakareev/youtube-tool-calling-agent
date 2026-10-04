"""The lab's first section: summarize a video by driving the tool calls by hand.

    python examples/manual_loop_step_by_step.py offline     # or gemini / openai
"""

import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from yt_agent import TOOL_MAPPING, TOOLS, execute_tool, get_llm

load_dotenv()
llm_with_tools = get_llm(sys.argv[1] if len(sys.argv) > 1 else None).bind_tools(TOOLS)

messages = [HumanMessage(content="I want to summarize youtube video: https://www.youtube.com/watch?v=SmplAI00001 in english")]
print("1. Human:", messages[0].content)

step = 2
while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    if not response.tool_calls:
        print(f"{step}. Final answer:\n{response.content}")
        break
    for call in response.tool_calls:
        print(f"{step}. Model asks for {call['name']}({call['args']})  [id={call['id']}]")
        tool_message = execute_tool(call)              # tool_mapping[name].invoke(args) -> ToolMessage
        messages.append(tool_message)
        preview = tool_message.content[:90] + ("…" if len(tool_message.content) > 90 else "")
        print(f"   ToolMessage(tool_call_id={call['id']}) = {preview}")
        step += 1

print("\nHistory:", " → ".join(type(m).__name__ for m in messages))
print("Tools available:", ", ".join(TOOL_MAPPING))
