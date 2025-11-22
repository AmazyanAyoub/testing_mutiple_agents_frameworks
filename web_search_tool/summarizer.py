from typing import List
from pydantic import BaseModel, Field
from .prompt import SUMMARY_WITH_SCORE_PROMPT
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from dotenv import load_dotenv
_= load_dotenv()

# choose whatever model you actually have in Ollama
OLLAMA_SUMMARIZER_MODEL = "llama3.2:3b"  # or "qwen2.5:7b", etc.
OLLAMA_SUMMARIZER_CTX = 128000

class SummaryScoreSchema(BaseModel):
    summary: str = Field(description="3-5 bullet summary as plain text")
    score: int = Field(description="Relevance 1-10 (10 = directly answers)")

def split_content_for_summarization(
    content: str,
    chunk_size: int = OLLAMA_SUMMARIZER_CTX,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Take one page's clean content string and return chunks
    ready to be summarized individually.
    """
    if not content:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # paragraphs
            "\n",     # line breaks
            ". ",     # sentence ends
            "? ",
            "! ",
            ", ",
        ],
    )
    return splitter.split_text(content)


async def summarize_single_content(content: str, question: str, llm):
    content = (content or "").strip()
    if not content:
        return {"summary": "content unavailable", "score": 0}

    prompt = SUMMARY_WITH_SCORE_PROMPT.replace("{question}", question).replace("{content}", content)
    res = await llm.ainvoke(prompt)  # plain chain: prompt | llm

    # Normalize res to text
    if hasattr(res, "content"):
        res_text = res.content
    else:
        res_text = str(res)

    try:
        data = json.loads(res_text)
    except Exception:
        # Try to recover from common wrappers
        try:
            data = json.loads(res_text.replace("```json", "").replace("```", "").strip())
        except Exception:
            return {"summary": res_text, "score": 0}

    # Unwrap schema-like shapes
    if isinstance(data, dict) and "properties" in data and isinstance(data["properties"], dict):
        data = data["properties"]

    summary_val = data.get("summary", "content unavailable")
    score_val = data.get("score", 0)

    # If summary/score are lists, take first; if score is str, cast
    if isinstance(summary_val, list) and summary_val:
        summary_val = " ".join(str(x) for x in summary_val)
    if isinstance(score_val, list) and score_val:
        score_val = score_val[0]
    try:
        score_val = int(score_val)
    except Exception:
        score_val = 0

    return {"summary": summary_val, "score": score_val}
