"""Utilities for building a LangGraph ReAct agent backed by Groq and MCP tools."""

from __future__ import annotations

import ast, asyncio, json, os
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from typing import Any, Dict, Iterable, List, Optional

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

from midleware_lanchain import retry_failed_tools, FullLoggingMiddleware, model_limit, reg_limit

load_dotenv()


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
MAX_WEB_RESULTS = 1


# CURRENT_TIME_SCHEMA: Dict[str, Any] = {
#     "type": "object",
#     "properties": {},
#     "required": [],
#     "description": "No arguments required.",
# }

CURRENT_TIME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Optional free-text reason for calling this tool. The model can put anything here.",
        },
    },
    "required": [],  # `reason` is optional
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
        description="Returns the current UTC time. Call it whenever the user asks for the current time, date, or 'now'.",
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

def _normalize_mcp_urls(urls: str | Iterable[str] | None = None) -> list[Connection]:
    if urls is None:
        env_url = os.getenv("MCP_URL")
        urls = ast.literal_eval(env_url)
        return [
            {"transport": "streamable_http", "url": u.strip()}
            for u in urls
            if u and u.strip()
        ]
        
    if isinstance(urls, str):
        urls = [urls]

    return [
        {"transport": "streamable_http", "url": u.strip()}
        for u in urls
        if u and u.strip()
    ]


async def gather_all_tools() -> List[BaseTool]:
    connections = _normalize_mcp_urls()
    mcp_tools = []
    for conn in connections:
        mcp_tools.extend(await load_mcp_tools(session=None, connection=conn))
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
        streaming=True
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
    "- MCP regulation tool (`lookup_regulation`): Fetch the official status/metadata for a regulation code (e.g., “AAMI TIR50”); call it at most once per unique code and then finalize the answer without calling tools again."
    "and trig problems. Invoke them for any numeric operation instead of calculating in your head.\n\n"
    "# condition question:\n"
    "For any “if/when/unless … then …” request: use tools to evaluate the condition; if TRUE, run the required action via the right tool(s) before answering; if FALSE, say why it wasn’t executed."
    "Don’t repeat identical tool calls; in the final answer, briefly report the condition result and (if applicable) the action outcome."
    "Example of a condition question and answer:  “add 4 and 5; if > 8 give current time” → call add(4,5)=9 (>8), then call current_time, then answer with both results."
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
    return create_agent(
        llm, 
        tools, 
        system_prompt=STRICT_TOOL_PROMPT, 
        middleware=[
            reg_limit,
            model_limit,
            FullLoggingMiddleware(),
            retry_failed_tools
    ])

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
        # 1) Token streaming from model
        if mode == "messages":
            token_chunk, _metadata = (
                chunk if isinstance(chunk, tuple) else (chunk, None)
            )
            if isinstance(token_chunk, AIMessageChunk) and token_chunk.content:
                yield {"text": token_chunk.content, "field": "response", "done": False}
            continue

        # 2) Node / tool updates
        # Sometimes `chunk` is not a dict, or values can be None
        if not isinstance(chunk, dict):
            # optional: print for debug
            # print("Non-dict update chunk:", chunk)
            continue

        for _, payload in chunk.items():
            if payload is None or not isinstance(payload, dict):
                # optional debug:
                # print("Skipping payload:", payload)
                continue

            msgs = payload.get("messages") or []
            if not msgs:
                continue

            last = msgs[-1]

            if isinstance(last, ToolMessage):
                name = getattr(last, "name", "") or getattr(last, "tool_name", "")
                if name and name not in seen_tools:
                    seen_tools.add(name)
                    yield {"text": name, "field": "used_tools", "done": False}

    # end-of-stream marker
    yield {"text": "", "field": "response", "done": True}



async def main():
    user_prompt = "what is the addition of 789 and 45 if the result > 790 give me the current time?"

    async for chunk in stream_response(user_prompt):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())