import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ensure package imports work when running from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_tool import run_search_with_summary  # noqa: E402


class DummyEmbedder:
    async def embed(self, texts):
        # returns a zero vector per text; scoring will just keep original order
        return {"embeddings": [[0.0] for _ in texts]}


def build_optimizer_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )


def build_summarizer_llm():
    # reuse the same Groq model for summaries; swap if you prefer Ollama, etc.
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


async def main():
    load_dotenv()

    optimizer_llm = build_optimizer_llm()
    summarizer_llm = build_summarizer_llm()
    embedder = DummyEmbedder()

    question = "latest news about reusable rockets"
    result = await run_search_with_summary(
        optimizer_llm=optimizer_llm,
        summarizer_llm=summarizer_llm,
        embedder=embedder,
        question=question,
        max_results=10,
        top_k=5,
        final_results=3,
        timelimit="m",
    )

    out_path = ROOT / "outputs" / "search_with_summaries.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
