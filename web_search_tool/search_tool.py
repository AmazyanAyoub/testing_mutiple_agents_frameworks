from typing import Any, Dict, List, Tuple, Optional
import os, json, asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from prompt import SEARCH_QUERY_PROMPT
from ddgs import DDGS
from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright
from filter_html import _extract_clean_text
from dotenv import load_dotenv
_= load_dotenv()


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
    output_path: str = "debug_search/web_result.json",
) -> Dict[str, Any]:
    
    chain = build_search_query_chain()
    schema: WebSearchQuerySchema = await chain.ainvoke({"question": question})
    optimized_query = schema.query

    search_raw = await _web_search_coroutine(optimized_query, max_results=max_results)
    if isinstance(search_raw, str):
        search_data = json.loads(search_raw)
    else:
        search_data = search_raw

    results: List[Dict[str, Any]] = search_data.get("results", [])
    urls = [r.get("url") for r in results if r.get("url")]

    fetch_tasks = [fetch_clean_page_content(url) for url in urls]
    fetched_pages = await asyncio.gather(*fetch_tasks)

    final_payload: Dict[str, Any] = {
        "question": question,
        "optimized_query": optimized_query,
        "search_results": results,
        "fetched_pages": fetched_pages,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return final_payload


if __name__ == "__main__":
    user_query = "zero-copy deserialization Rust -serde -python site:github.com"
    asyncio.run(run_search_and_fetch(question=user_query))