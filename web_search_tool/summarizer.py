from typing import Any, Dict, List
from langchain_ollama import ChatOllama
from prompt import PAGE_SUMMARY_PROMPT, REDUCE_SUMMARY_PROMPT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
_= load_dotenv()

# choose whatever model you actually have in Ollama
OLLAMA_SUMMARIZER_MODEL = "llama3.2:3b"  # or "qwen2.5:7b", etc.
OLLAMA_SUMMARIZER_CTX = 8192

def get_ollama_summarizer_llm() -> Any:
    """
    Return the Ollama chat model used for per-page summarization.
    """
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=OLLAMA_SUMMARIZER_MODEL,
        temperature=0.1,   # stable summaries
    )
    return llm

def split_content_for_summarization(
    content: str,
    chunk_size: int = 8192,
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

def build_page_summary_chain() -> Any:
    llm = get_ollama_summarizer_llm()
    parser = StrOutputParser()
    chunk_prompt = ChatPromptTemplate.from_template(PAGE_SUMMARY_PROMPT.strip())
    reduce_prompt = ChatPromptTemplate.from_template(REDUCE_SUMMARY_PROMPT.strip())
    return {
        "chunk": chunk_prompt | llm | parser,
        "reduce": reduce_prompt | llm | parser,
    }


async def _reduce_summaries(
    chain: Any,
    summaries: List[str],
    batch_size: int = 5,
) -> str:
    """
    Iteratively merge partial summaries in bounded batches so that the
    reducer prompt never receives an enormous wall of text.
    """
    current = [s.strip() for s in summaries if s.strip()]
    if not current:
        return ""

    while len(current) > 1:
        next_round: List[str] = []
        for i in range(0, len(current), batch_size):
            block = "\n\n".join(current[i : i + batch_size])
            merged = await chain.ainvoke({"content": block})
            next_round.append(merged.strip())
        current = next_round

    return current[0]


async def summarize_single_content(content: str, question: str) -> str:
    content = (content or "").strip()
    if not content:
        return ""

    chains = build_page_summary_chain()
    chunk_chain = chains["chunk"]

    if len(content) <= OLLAMA_SUMMARIZER_CTX:
        return (await chunk_chain.ainvoke({"content": content, "question": question})).strip()

    chunk_size = max(1500, OLLAMA_SUMMARIZER_CTX - 512)
    chunks = split_content_for_summarization(content, chunk_size=chunk_size)
    if not chunks:
        return ""

    partials = [
        (await chunk_chain.ainvoke({"content": chunk, "question": question})).strip()
        for chunk in chunks
    ]
    return "\n\n".join(partials)