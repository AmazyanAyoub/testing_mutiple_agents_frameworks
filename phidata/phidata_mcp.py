import asyncio
import json
import os
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools import Function, tool
from phi.tools.duckduckgo import DuckDuckGo
from pydantic import BaseModel, field_validator

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

load_dotenv()

MCP_COMMAND = os.getenv("MCP_COMMAND", "python")
MCP_SCRIPT = os.getenv("MCP_SCRIPT", "mcp_server.py")
MCP_SCRIPT_PATH = os.path.abspath(MCP_SCRIPT)


def weather(city: str):
    """Use this tool to return the weather of a given city"""
    answer = f"Weather in {city}: 45 C sunny"
    return json.dumps(answer)


def joke():
    """Use this tool to give a joke"""
    response = "Why did the AI cross the road? Because GPUs were cheaper on the other side."
    return json.dumps(response)


# @tool(name="duckduckgo_search")
# def duckduckgo_search(query: str, max_results: int = 5) -> str:
#     """Search DuckDuckGo and return JSON snippets."""
#     if DDGS is None:
#         return "Install duckduckgo_search to enable this tool."

#     # clamp to a reasonable size
#     max_results = max(1, min(max_results, 10))
#     results = []
#     try:
#         with DDGS() as ddg:
#             for item in ddg.text(query, max_results=max_results):
#                 results.append(
#                     {
#                         "title": item.get("title", ""),
#                         "href": item.get("href", ""),
#                         "body": item.get("body", ""),
#                     }
#                 )
#     except Exception as exc:
#         return f"DuckDuckGo search failed: {exc}"

#     if not results:
#         return f"No results found for {query!r}"

#     return json.dumps(results, ensure_ascii=False)

class AddRequest(BaseModel):
    a: float
    b: float

    @field_validator("a", "b")
    @classmethod
    def finite(cls, v: float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be finite number")
        return v


class AddResponse(BaseModel):
    result: float


@tool
def add_strict(a: float, b: float) -> str:
    req = AddRequest(a=a, b=b)
    res = AddResponse(result=req.a + req.b)
    return res.model_dump_json()


async def _load_tools_via_mcp(server_params: StdioServerParameters) -> List[BaseTool]:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await load_mcp_tools(session)


def load_mcp_tools_sync(server_params: StdioServerParameters):
    return _run_coroutine_safely(lambda: _load_tools_via_mcp(server_params))


def _extract_json_schema(args_schema: Any) -> Dict[str, Any]:
    default_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    if args_schema is None:
        return default_schema

    schema: Optional[Dict[str, Any]] = None
    if isinstance(args_schema, dict):
        schema = args_schema
    elif isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        schema = args_schema.model_json_schema()

    if not schema:
        return default_schema

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    return {"type": "object", "properties": properties, "required": required}


def _run_coroutine_safely(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    if not loop.is_running():
        return asyncio.run(coro_factory())

    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result_holder["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001
            error_holder["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_holder:
        raise error_holder["error"]

    return result_holder.get("value")


def _format_tool_output(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
        content_text = _format_tool_output(content)
        artifact_text = _format_tool_output(artifact)
        if content_text and artifact_text:
            return f"{content_text}\n\nArtifact: {artifact_text}"
        return content_text or artifact_text

    if isinstance(result, BaseMessage):
        return _format_tool_output(result.content)

    if isinstance(result, list):
        if all(isinstance(item, str) for item in result):
            return "\n".join(item for item in result if item)
        try:
            return json.dumps(result, default=str)
        except TypeError:
            return "\n".join(str(item) for item in result)

    if isinstance(result, dict):
        try:
            return json.dumps(result, indent=2, default=str)
        except TypeError:
            return str(result)

    return str(result)


def convert_langchain_tool_to_phi(tool: BaseTool) -> Function:
    schema = _extract_json_schema(getattr(tool, "args_schema", None))

    def wrapper(agent: Any = None, **kwargs: Any) -> str:
        if agent is not None:
            kwargs.pop("agent", None)

        def coro_factory() -> Any:
            return tool.ainvoke(kwargs)

        result = _run_coroutine_safely(coro_factory)
        return _format_tool_output(result)

    return Function(
        name=tool.name,
        description=tool.description or "",
        parameters=schema,
        entrypoint=wrapper,
    )


_ddgs = DDGS()
_ddgs_cache: dict[tuple[str, int], dict[str, Any]] = {}


def brave_search(query: str, max_results: int = 2) -> str:
    """Use this function to search the public web and answer that are general via DuckDuckGo.

    Args:
        query (str): The search phrase.
        max_results (int): Maximum number of results (1–10). Defaults to 2.

    Returns:
        str: JSON string containing results or an error payload.
    """
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "empty query"})

    max_results = max(1, min(int(max_results), 10))
    cache_key = (query, max_results)
    if cache_key in _ddgs_cache:
        return json.dumps(_ddgs_cache[cache_key], ensure_ascii=False)

    try:
        hits = []
        with _ddgs as ddg:
            for item in ddg.text(query, max_results=max_results) or []:
                hits.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
        payload = {
            "query": query,
            "results": hits,
            "meta": {"source": "ddgs", "result_count": len(hits)},
        }
    except Exception as exc:  # noqa: BLE001
        payload = {"query": query, "error": str(exc)}

    _ddgs_cache[cache_key] = payload
    return json.dumps(payload, ensure_ascii=False)

SYSTEM_PROMPT = """
You must use available tools to answer the user.

Tools:
- weather(city): returns a demo weather report.
- joke(): tells a canned joke.
- brave_search(query, max_results=2): searches the web via DuckDuckGo.
- (If MCP tools are present) Use each MCP tool exactly as named and described.

Workflow:
1) Pick the best tool for the request.
2) Call it.
3) Summarize the tool result for the user.
If none apply, reply exactly: "I don't have a tool for that."
"""

if __name__ == "__main__":
    print("loading MCP tools ...")
    server_params = StdioServerParameters(
        command=MCP_COMMAND,
        args=[MCP_SCRIPT_PATH],
    )
    mcp_langchain_tools = load_mcp_tools_sync(server_params)
    print(f"[OK] Loaded {len(mcp_langchain_tools)} MCP tools")

    phi_mcp_tools = [convert_langchain_tool_to_phi(tool) for tool in mcp_langchain_tools]
    all_tools = [weather, joke, brave_search] + phi_mcp_tools

    agent = Agent(
        model=Groq(id="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")),
        # system_prompt=SYSTEM_PROMPT,
        instructions=[SYSTEM_PROMPT],
        tools=all_tools,
        show_tool_calls=True,
        markdown=True,
    )

    while True:
        try:
            q = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break

        result = agent.run(q, stream=False)
        print("Answer:", result.content)

        # print("Tool calls:")
        # for call in result.tool_calls or []:
        #     print(f"  {call.name} -> {call.arguments}")