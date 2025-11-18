from datetime import datetime
import json, asyncio
from typing import Any, Dict
from pathlib import Path

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None

MAX_WEB_RESULTS = 10

WEB_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search keywords."},
        "max_results": {
            "type": "integer",
            "description": f"Number of top hits to return (1-{MAX_WEB_RESULTS}).",
            "default": 10,
        },
    },
    "required": ["query"],
}


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
                        "snippet": item.get("body", "")
                    }
                )
        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    payload = await loop.run_in_executor(None, _do_search)
    # --- Debug save (best-effort; never break the API) ---
    try:
        debug_dir = Path("debug_search")
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
        safe_query = (safe_query or "empty")[:60]
        debug_path = debug_dir / f"ddgs_{safe_query}_{ts}.json"
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return json.dumps(payload, ensure_ascii=False)


HARD_QUERIES = [
    # 1) Deep tech spec + phrase search + negative filters
    '"zero-copy deserialization" Rust -serde -python site:github.com',

    # 2) Scientific + site restriction + filetype
    'site:arxiv.org "implicit neural representation" filetype:pdf "comparative analysis"',

    # 3) Legal/regulation with exact phrase and minus noise
    '"right to be forgotten" GDPR comparative jurisprudence -wikipedia',

    # 4) Security CVEs with operator mix
    'inurl:CVE "heap overflow" intitle:advisory site:github.com OR site:lists.debian.org',

    # 5) Multi-lingual query (French) + exact phrase + site scope
    'site:legifrance.gouv.fr "intelligence artificielle" "responsabilité" "données personnelles"',

    # 6) Data engineering + tricky operators
    '"idempotent data pipelines" best practices -spark -airflow filetype:pdf',

    # 7) Niche ML topic + site scope
    'site:paperswithcode.com "long-context" retrieval "needles in a haystack"',

    # 8) Rare phrase across academic + negative terms
    '"symbolic regression" benchmarks "generalization" -marketing -sponsored',

    # 9) Non-English (Arabic) + exact phrase
    'تعلم الآلة "نماذج اللغة الكبيرة" التقييم المقارن',

    # 10) Product internals + inurl/intitle combos
    'inurl:docs intitle:"rate limits" "exponential backoff" "429" "best practices"',

    # 11) Government datasets + phrases
    'site:data.gov "road traffic" "accident severity" "open data"',

    # 12) Rare software pattern + negative filters
    '"event sourcing" pitfalls "exactly-once" -kafka -confluent',

    # 13) Cryptography topic with site filter
    'site:ietf.org "hybrid public key encryption" draft filetype:pdf',

    # 14) Multi keyword logic + quotes
    '"streaming vector search" ANN "HNSW" "real-time" -advertisement',

    # 15) DevOps edge case
    '"blue-green deployment" "database migrations" "zero downtime" best practices',
]


async def _run_one(query: str, max_results: int = 10) -> dict:
    payload_str = await _web_search_coroutine(query, max_results=max_results)
    try:
        data = json.loads(payload_str)
    except json.JSONDecodeError:
        data = {"query": query, "result_count": 0, "results": [], "error": "invalid_json_from_coroutine", "raw": payload_str}
    # keep only what we need
    return {
        "query": data.get("query", query),
        "result_count": data.get("result_count", len(data.get("results", []))),
        "results": data.get("results", []),
        **({"error": data["error"]} if "error" in data else {})
    }

async def main() -> None:
    combined = {}
    for i, q in enumerate(HARD_QUERIES, start=1):
        combined[f"question_{i}"] = await _run_one(q, max_results=10)

    debug_dir = Path("debug_search")
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "ddgs_all.json"  # fixed name, overwritten every run
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Saved ONE file with everything to: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
