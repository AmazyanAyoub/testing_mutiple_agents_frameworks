import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

SUMMARIZER_MODEL = os.getenv("OLLAMA_SUMMARIZER_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_ENDPOINT = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# ensure package imports work when running from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_tool import run_search_with_summary  # noqa: E402
from embedder import OllamaEmbedder

def build_optimizer_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )


def build_summarizer_llm():
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=SUMMARIZER_MODEL,
        temperature=0.1,
        num_ctx=8192,  # adjust to your model’s context window
    )


def build_embedder():
    return OllamaEmbedder(
        model_name=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

async def main():
    load_dotenv()

    optimizer_llm = build_optimizer_llm()
    summarizer_llm = build_summarizer_llm()
    embedder = build_embedder()

    question = "can you tell the latest news about generative ai and what the world has achieved?"
    result = await run_search_with_summary(
        optimizer_llm=optimizer_llm,
        summarizer_llm=summarizer_llm,
        embedder=embedder,
        question=question,
        max_results=20,
        top_k=5,
        final_results=3,
        timelimit="m",
        mode="custom"
    )

    out_path = ROOT / "outputs" / "search_with_summaries.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
