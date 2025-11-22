from typing import Any, Dict, List, Optional
from ddgs import DDGS

VALID_TIME_LIMITS = {"d", "w", "m", "y"}

def ddgs_search(
    query: str,
    *,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timelimit: Optional[str] = "m",
) -> Dict[str, Any]:
    """
    Run a DuckDuckGo text search and return a normalized payload.
    timelimit: None or one of {"d", "w", "m", "y"} for day/week/month/year.
    Returns:
      {
        "query": str,
        "result_count": int,
        "results": [
          {"title": str, "url": str, "snippet": str},
          ...
        ],
        "error": Optional[str]
      }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "query": query,
            "result_count": 0,
            "results": [],
            "error": "query must not be empty",
        }

    tlimit = timelimit if timelimit in VALID_TIME_LIMITS else None
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddg:
            for item in ddg.text(
                query.strip(),
                max_results=max_results,
                region=region,
                safesearch=safesearch,
                timelimit=tlimit,
            ) or []:
                results.append(
                    {
                        "title": item.get("title", "") or "",
                        "url": item.get("href", "") or "",
                        "snippet": item.get("body", "") or "",
                    }
                )
        return {
            "query": query,
            "result_count": len(results),
            "results": results,
            "error": None,
        }
    except Exception as exc:
        return {
            "query": query,
            "result_count": 0,
            "results": [],
            "error": str(exc),
        }
