import asyncio, json, logging, os, math, time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .prompt import SEARCH_QUERY_PROMPT
from .filter_html import _extract_clean_text
from .summarizer import summarize_single_content
from provider.ddgs_provider import ddgs_search
from provider.playwright_scraper import fetch_page_html

load_dotenv()

logger = logging.getLogger(__name__)

class WebSearchQuerySchema(BaseModel):
    query: str = Field(
        ...,
        description="Optimized search query string to send to the web search engine.",
    )

def build_search_query_chain(llm) -> Any:
    parser = PydanticOutputParser(pydantic_object=WebSearchQuerySchema)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate.from_template(
        SEARCH_QUERY_PROMPT.strip()
    ).partial(format_instructions=format_instructions)

    return prompt | llm | parser

async def _run_ddgs(query: str, max_results: int, timelimit: Optional[str]) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: ddgs_search(query=query, max_results=max_results, timelimit=timelimit),
    )

async def _fetch_pages(urls: List[str], timeout_ms: int = 30000) -> List[Dict[str, Any]]:
    tasks = [fetch_page_html(url, timeout_ms=timeout_ms) for url in urls]
    return await asyncio.gather(*tasks) if tasks else []


async def score_results(
    query: str,
    results: List[Dict[str, Any]],
    embedder: Any,              # expects .embed(texts: List[str]) -> List[List[float]]
    top_k: int = 5,
    min_score: float = 0.25,    # drop obvious off-topic hits
) -> List[Dict[str, Any]]:
    """
    Compute cosine similarity between query and each result's title+snippet.
    Returns the top_k results with an added 'similarity' field, sorted desc.
    Falls back to original ordering on any embedding error.
    """
    if not results:
        return results

    try:
        query_vec = (await embedder.embed([query]))["embeddings"][0]
        qnorm = math.sqrt(sum(x * x for x in query_vec)) or 1.0

        texts = [(r.get("title", "") + " " + r.get("snippet", "")).strip() for r in results]
        doc_vecs = (await embedder.embed(texts))["embeddings"]

        rescored: List[Dict[str, Any]] = []
        for r, dv in zip(results, doc_vecs):
            dnorm = math.sqrt(sum(x * x for x in dv)) or 1.0
            sim = sum(a * b for a, b in zip(query_vec, dv)) / (qnorm * dnorm)
            # if sim >= min_score:
            rescored.append({**r, "similarity": sim})

        rescored.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        if top_k:
            rescored = rescored[: min(top_k, len(rescored))]
        return rescored or results  # if all filtered out, return originals
    except Exception as e:
        print(e)
        return results  # on embedding failure, do not block the pipeline

async def run_search_and_fetch(
    question: str,
    optimizer_llm: Any,
    embedder: Any,
    max_results: int = 20,
    top_k: int = 5,
    timelimit: Optional[str] = "m",
    fetch_timeout_ms: int = 30000,
) -> Dict[str, Any]:
    """
    Optimize the query, run DDGS search, and fetch page HTML.
    Returns:
      {
        "question": ...,
        "optimized_query": ...,
        "search_results": [...],
        "fetched_pages": [...]
      }
    """
    if not question or not question.strip():
        return {"error": "question must not be empty", "question": question}
    t0 = time.monotonic()
    chain = build_search_query_chain(optimizer_llm)
    schema: WebSearchQuerySchema = await chain.ainvoke({"question": question})
    optimized_query = schema.query.strip()
    t_opt = time.monotonic()

    search_payload = await _run_ddgs(optimized_query, max_results=max_results, timelimit=timelimit)
    results = search_payload.get("results") or []
    t_ddg = time.monotonic()
    results = await score_results(query=question, results=results, embedder=embedder, top_k=top_k, min_score=0.25) 
    t_score = time.monotonic()
    urls = [entry["url"] for entry in results if entry.get("url")]

    fetched = await _fetch_pages(urls, timeout_ms=fetch_timeout_ms)
    t_fetch = time.monotonic()
    cleaned_pages: List[Dict[str, Any]] = []
    for page in fetched:
        html = page.get("html", "")
        clean_text = _extract_clean_text(html) if html else ""
        cleaned_pages.append({**page, "clean_text": clean_text})
    timings = {
    "total": time.monotonic() - t0,
    "optimize": t_opt - t0,
    "search": t_ddg - t_opt,
    "fetch_clean": t_fetch - t_ddg,
    "embed_score": t_score - t_ddg  # if you time score_results
    }

    return {
        "question": question,
        "optimized_query": optimized_query,
        "search_results": results,
        "fetched_pages": fetched,
        "fetched_pages": cleaned_pages,
        "search_error": search_payload.get("error"),
        "timings": timings,
    }

async def run_search_with_summary(
    optimizer_llm: Any,
    summarizer_llm: Any,
    embedder: Any,
    question: str,
    max_results: int = 20,
    top_k: int = 5,
    final_results: int = 3,
    timelimit: Optional[str] = "m",
    fetch_timeout_ms: int = 30000,
) -> Dict[str, Any]:
    
    t0 = time.monotonic()
    
    base = await run_search_and_fetch(
        question=question,
        max_results=max_results,
        embedder=embedder,
        timelimit=timelimit,
        fetch_timeout_ms=fetch_timeout_ms,
        optimizer_llm=optimizer_llm,
        top_k=top_k, 
    )

    timings = dict(base.get("timings", {}))
    title_lookup = {
        entry.get("url"): entry.get("title") or entry.get("url")
        for entry in base.get("search_results", [])
    }

    scores: List[Dict[str, Any]] = []
    summary_error = None
    t_sum_start = time.monotonic()
    try:
        for page in base.get("fetched_pages", []):
            url = page.get("url")
            content = page.get("clean_text", "")
            summary_obj = await summarize_single_content(content, question, summarizer_llm)
            summary_text = summary_obj.get("summary") if isinstance(summary_obj, dict) else str(summary_obj)
            score_val = summary_obj.get("score") if isinstance(summary_obj, dict) else None

            scores.append(
                {
                    "title": title_lookup.get(url, url),
                    "url": url,
                    "snippets": summary_text,
                    "score": score_val,
                }
            )

        # sort by score desc and keep top 3
        scores.sort(key=lambda x: x.get("score", 0), reverse=True)
        scores = scores[: min(final_results, len(scores))]

    except Exception as exc:
        summary_error = str(exc)

    t_sum_end = time.monotonic()
    timings["summaries"] = t_sum_end - t_sum_start
    timings["total"] = time.monotonic() - t0

    return {
        "query": question,
        "optimized_query": base.get("optimized_query"),
        "result_count": len(scores),
        "results": scores,
        "search_results": base.get("search_results"),
        "fetched_pages": base.get("fetched_pages"),
        "search_error": base.get("search_error"),
        "summary_error": summary_error,
        "timings": timings,
    }

