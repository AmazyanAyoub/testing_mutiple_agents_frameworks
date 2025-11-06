import json, asyncio
from typing import Any, Dict
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None


MAX_WEB_RESULTS = 3

async def _web_search_coroutine(query: str, max_results: int = 3) -> str:
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



if __name__ == "__main__":
    user_prompt = "who is donald trump"
    res = asyncio.run(_web_search_coroutine(user_prompt))
    print(res)
