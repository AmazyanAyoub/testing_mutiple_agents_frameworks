from typing import Any, Dict, List, Tuple, Optional
import os, json, asyncio, logging, time
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from ddgs import DDGS
from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright
from dotenv import load_dotenv
_= load_dotenv()


import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from filter_html import _extract_clean_text
from web_search_tool.summarizer import summarize_single_content
from prompt import SEARCH_QUERY_PROMPT

for noisy in ("httpx", "httpcore", "urllib3", "playwright.async_api"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class WebSearchQuerySchema(BaseModel):
    query: str = Field(
        ...,
        description="Optimized search query string to send to the web search engine.",
    )

def build_search_query_chain() -> Any:
    parser = PydanticOutputParser(pydantic_object=WebSearchQuerySchema)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate.from_template(
        SEARCH_QUERY_PROMPT.strip()
    ).partial(format_instructions=format_instructions)

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    return prompt | llm | parser



async def _web_search_coroutine(query: str, max_results: int = 5) -> str:
    if not query or not query.strip():
        return json.dumps({"error": "query must not be empty"})
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

async def fetch_clean_page_content(url: str, timeout_ms: int = 30000) -> Dict[str, Any]:
    if not url or not url.strip():
        return {
            "url": url,
            "status": None,
            "error": "url must not be empty",
            "content": "",
        }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()

            async def _block_heavy(route, request):
                if request.resource_type in {"image", "media", "font", "stylesheet"}:
                    await route.abort()
                else:
                    await route.continue_()
                    
            await page.route("**/*", _block_heavy)
            response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = await page.content()
            clean_text = _extract_clean_text(html)

            status = response.status if response is not None else None

            return {
                "url": url,
                "status": status,
                "error": None if status and 200 <= status < 400 else f"HTTP {status}",
                "content": clean_text,
            }
        except Exception as exc:
            return {
                "url": url,
                "status": None,
                "error": str(exc),
                "content": "",
            }
        finally:
            await browser.close()


async def run_search_and_fetch(
    question: str,
    max_results: int = 5,
) -> Dict[str, Any]:
    print("run_search_and_fetch:start question=%s", question)

    chain = build_search_query_chain()
    schema: WebSearchQuerySchema = await chain.ainvoke({"question": question})
    optimized_query = schema.query.strip()
    print("Optimized query: %s", optimized_query)

    search_raw = await _web_search_coroutine(optimized_query, max_results=max_results)
    search_data = json.loads(search_raw) if isinstance(search_raw, str) else (search_raw or {})
    results = search_data.get("results") or []
    urls = [entry["url"] for entry in results if entry.get("url")]
    print("Collected %d URLs for fetching", len(urls))

    fetched_pages = await asyncio.gather(*(fetch_clean_page_content(url) for url in urls)) if urls else []

    payload = {
        "question": question,
        "optimized_query": optimized_query,
        "search_results": results,
        "fetched_pages": fetched_pages,
    }

    print("run_search_and_fetch:end question=%s", question)
    return payload


async def run_search_with_summary(
    question: str,
    max_results: int = 5,
    # output_path: str = "debug_search/page_summaries.json",
) -> Dict[str, Any]:
    print("run_search_with_summary:start question=%s", question)

    base_payload = await run_search_and_fetch(
        question=question,
        max_results=max_results
    )

    print("Summarizing fetched page contents")
    title_lookup = {
        entry.get("url"): entry.get("title") or entry.get("url")
        for entry in base_payload.get("search_results", [])
    }

    summaries: List[Dict[str, Any]] = []
    for page in base_payload.get("fetched_pages", []):
        url = page.get("url")
        clean_summary = await summarize_single_content(page.get("content", ""), question)
        summaries.append(
            {
                "title": title_lookup.get(url, url),
                "url": url,
                "snippets": clean_summary or "NOT_RELEVANT",
            }
        )

    payload = {
        "query":question,
        "result_count": len(summaries),
        "results": summaries,
    }

    # debug_path = Path("debug_search/page_summaries.json")
    # debug_path.parent.mkdir(parents=True, exist_ok=True)
    # debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("run_search_with_summary:end question=%s", question)
    return payload


if __name__ == "__main__":
    user_query = "zero-copy deserialization Rust -serde -python site:github.com"
    start_time = time.time()
    asyncio.run(run_search_with_summary(question=user_query))
    duration = time.time() - start_time
    print(f"Completed in {duration:.2f} seconds")