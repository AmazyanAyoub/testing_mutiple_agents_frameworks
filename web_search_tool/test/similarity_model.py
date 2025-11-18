# similarity_model_check.py
# pip install sentence-transformers numpy

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("debug_search/ddgs_all.json")
OUTPUT_PATH = Path("debug_search/similarity_report_model.json")

# You can switch to another small, fast model if you want
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = 0.40  # tweak as you like

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _combine_text(res_item: Dict[str, str]) -> str:
    title = (res_item.get("title") or "").strip()
    snippet = (res_item.get("snippet") or "").strip()
    body = (res_item.get("body") or "").strip()
    # avoid duplicating snippet==body
    if snippet and body and snippet == body:
        body = ""
    combined = " ".join([title, snippet, body]).strip()
    return combined or title or snippet or body

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # a: (d,), b: (d,)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _score_question(model: SentenceTransformer, query: str, results: List[Dict]) -> Tuple[List[float], List[Dict]]:
    """
    Encode query and each result (title+snippet+body) with SentenceTransformer,
    return per-result cosine similarities and detailed rows.
    """
    # Build corpus: first element is query, rest are result texts
    corpus = [query] + [_combine_text(r) for r in results]
    # Batch encode for speed
    embs = model.encode(corpus, batch_size=32, normalize_embeddings=True)
    q_emb = embs[0]
    r_embs = embs[1:]

    scores: List[float] = []
    detailed: List[Dict] = []
    for idx, (r, r_vec) in enumerate(zip(results, r_embs), start=1):
        s = _cosine(q_emb, r_vec)
        scores.append(s)
        detailed.append({
            "rank": idx,
            "title": r.get("title", ""),
            "url": r.get("url", r.get("href", "")),
            "score": round(s, 4),
            "snippet": r.get("snippet", "") or r.get("body", "")
        })
    return scores, detailed

def run_similarity_model(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    model_name: str = MODEL_NAME,
    threshold: float = THRESHOLD
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    data = _read_json(input_path)
    model = SentenceTransformer(model_name)

    # Expecting:
    # {
    #   "question_1": { "query": "...", "result_count": N, "results": [...] },
    #   ...
    # }
    question_keys = sorted(
        [k for k in data.keys() if k.startswith("question_")],
        key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 10**9,
    )

    results_out = {
        "model": model_name,
        "threshold": threshold,
        "summary": {},
        "questions": []
    }

    top_scores = []
    above_cnt = 0

    for key in question_keys:
        block = data.get(key) or {}
        query = (block.get("query") or "").strip()
        results = block.get("results") or []

        scores, detailed = _score_question(model, query, results)
        top = max(scores) if scores else 0.0
        avg = float(np.mean(scores)) if scores else 0.0
        is_good = top >= threshold

        top_scores.append(top)
        if is_good:
            above_cnt += 1

        results_out["questions"].append({
            "id": key,
            "query": query,
            "result_count": len(results),
            "top_score": round(top, 4),
            "avg_score": round(avg, 4),
            "above_threshold": is_good,
            "scores": detailed  # per-result rows with cosine scores
        })

    overall_avg_top = float(np.mean(top_scores)) if top_scores else 0.0
    results_out["summary"] = {
        "total_questions": len(question_keys),
        "avg_top_score": round(overall_avg_top, 4),
        "num_above_threshold": above_cnt
    }

    _write_json(output_path, results_out)
    print(f"Similarity (model) report saved to: {output_path.resolve()}")

if __name__ == "__main__":
    run_similarity_model()
