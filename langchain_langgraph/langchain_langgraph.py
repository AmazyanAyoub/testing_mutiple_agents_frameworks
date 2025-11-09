"""Utilities for building a LangGraph ReAct agent backed by Groq and MCP tools."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from typing import Any, Dict, List

from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, StructuredTool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None


load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
MAX_WEB_RESULTS = 1


CURRENT_TIME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}

WEB_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search keywords."},
        "max_results": {
            "type": "integer",
            "description": f"Number of top hits to return (1-{MAX_WEB_RESULTS}).",
            "default": 5,
        },
    },
    "required": ["query"],
}


async def _current_time_coroutine() -> str:
    now = datetime.now(timezone.utc)
    payload = {"iso_time": now.isoformat(), "timezone": "UTC"}
    return json.dumps(payload)


async def _web_search_coroutine(query: str, max_results: int = 1) -> str:
    if not query or not query.strip():
        return json.dumps({"error": "query must not be empty"})

    if DDGS is None:
        return json.dumps(
            {
                "error": "duckduckgo_search package is not installed",
                "hint": "pip install duckduckgo_search",
            }
        )

    try:
        max_results = max(1, min(int(max_results), MAX_WEB_RESULTS))
    except (TypeError, ValueError):
        max_results = 5

    loop = asyncio.get_running_loop()

    def _do_search() -> Dict[str, Any]:
        results = []
        with DDGS() as ddg:
            for item in ddg.text(query, max_results=max_results) or []:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    payload = await loop.run_in_executor(None, _do_search)
    return json.dumps(payload, ensure_ascii=False)


builtin_tools = [
    StructuredTool(
        name="current_time",
        description="Return the current UTC time in ISO 8601 format.",
        args_schema=CURRENT_TIME_SCHEMA,
        coroutine=_current_time_coroutine,
    ),
    StructuredTool(
        name="web_search",
        description="Search DuckDuckGo and return a JSON payload of the top results.",
        args_schema=WEB_SEARCH_SCHEMA,
        coroutine=_web_search_coroutine,
    ),
]


def _get_mcp_server_url() -> str:
    """Read the MCP server URL from the environment."""
    server_url = os.getenv("MCP_URL")
    if not server_url:
        raise RuntimeError(
            "MCP_URL environment variable is not set. "
            "Update your .env file with MCP_URL pointing to the MCP HTTP endpoint."
        )
    return server_url.strip()

def _get_mcp_connection() -> Connection:
    """Build the connection parameters used for MCP tool access."""
    return {
        "transport": "streamable_http",
        "url": _get_mcp_server_url(),
    }


async def gather_all_tools() -> List[BaseTool]:
    connection = _get_mcp_connection()
    mcp_tools = await load_mcp_tools(session=None, connection=connection)
    return [*builtin_tools, *mcp_tools]


def _build_chat_model(model_name: str | None = None) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set; please export it or add it to your .env file."
        )

    return ChatGroq(
        model_name=model_name or DEFAULT_GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.0,
        streaming=True,
    )

STRICT_TOOL_PROMPT = (
    "You can always solve tasks by calling the available tools first, then summarizing "
    "their outputs in plain language. Follow these rules rigorously:\n\n"
    "- `current_time()`: Returns the current UTC timestamp. Use it whenever the user asks about “now,” "
    "requests time-sensitive context, or needs to know when an event is happening.\n"
    "- `web_search(query, max_results)`: Searches the public web (DuckDuckGo) for real-world information. "
    "Call it for news, biographies, historical facts, stats, or anything requiring up-to-date knowledge. "
    "Always read the snippets before answering.\n"
    "- MCP math tools (`add`, `subtract`, `multiply`, `divide`, `sin`): These are authoritative for arithmetic "
    "and trig problems. Invoke them for any numeric operation instead of calculating in your head.\n\n"
    "Workflow:\n"
    "1. Decide which tool (or sequence of tools) best supports the request.\n"
    "2. Invoke tool(s) and inspect their outputs.\n"
    "3. Compose a final answer that faithfully summarizes the tool results. "
    "If no tool helps, reply: “I cannot help with that using the current tools.”\n\n"
    "Do not fabricate information or answer without consulting the relevant tool."
)
async def build_react_agent(model_name: str | None = None):
    llm = _build_chat_model(model_name)
    tools = await gather_all_tools()
    return create_agent(llm, tools, system_prompt=STRICT_TOOL_PROMPT)

async def run_agent(prompt_text: str, model_name: str | None = None) -> dict[str, object]:
    agent = await build_react_agent(model_name)
    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt_text)]})
    messages = result.get("messages", [])

    answer = next(
        (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
        str(result),
    )
    tool_calls = [
        {"tool": msg.name, "output": msg.content}
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]
    return {"answer": answer, "tool_calls": tool_calls}


async def stream_response(
    prompt_text: str, model_name: str | None = None
) -> AsyncIterator[dict[str, object]]:
    """Stream tools + final text; tokens via 'messages', steps via 'updates'."""
    agent = await build_react_agent(model_name)
    seen_tools: set[str] = set()

    async for mode, chunk in agent.astream(
        {"messages": [HumanMessage(content=prompt_text)]},
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            token_chunk, _metadata = chunk if isinstance(chunk, tuple) else (chunk, None)
            if isinstance(token_chunk, AIMessageChunk) and token_chunk.content:
                yield {"text": token_chunk.content, "field": "response", "done": False}
            continue

        for _, payload in chunk.items():
            msgs = payload.get("messages") or []
            if not msgs:
                continue
            last = msgs[-1]

            if isinstance(last, ToolMessage):
                name = getattr(last, "name", "") or getattr(last, "tool_name", "")
                if name and name not in seen_tools:
                    seen_tools.add(name)
                    yield {"text": name, "field": "used_tools", "done": False}
                continue

    yield {"text": "", "field": "response", "done": True}


async def _preview_stream(prompt: str):
    printed_response = False
    async for event in stream_response(prompt):
        text = event["text"]
        if not text:
            continue
        field = event.get("field")
        if field == "used_tools":
            print(f"[used_tools] {text}")
            continue
        if field == "response":
            print(text, end="", flush=True)
            printed_response = True
    if printed_response:
        print()

if __name__ == "__main__":
    user_prompt = "add 789 to 1546"
    asyncio.run(_preview_stream(user_prompt))
