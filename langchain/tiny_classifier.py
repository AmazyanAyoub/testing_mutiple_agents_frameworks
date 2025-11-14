from typing import List, Optional
from pydantic import BaseModel, Field
import json, hashlib  # ADD
from langchain_core.messages import SystemMessage, HumanMessage  # ADD
from langchain_groq import ChatGroq  # ADD

from dotenv import load_dotenv
_ = load_dotenv()

ALLOWED_OPS = {">","<",">=","<=","==","!=","contains","not_contains"}

class Predicate(BaseModel):
    normalized: str
    lhs: Optional[str] = None
    op: Optional[str] = Field(None, pattern="|".join(map(repr, ALLOWED_OPS)))
    rhs: Optional[str] = None
    needs_tools: List[str] = []
    confidence: float = 0.0

class Action(BaseModel):
    normalized: str
    verbs: List[str] = []
    objects: List[str] = []
    needs_tools: List[str] = []
    confidence: float = 0.0

class ConditionalIntent(BaseModel):
    has_conditional: bool
    language: str
    predicate: Predicate
    action: Action
    tool_hints: List[str] = []
    confidence: float = 0.0


# ADD — tiny classifier prompts
CLASSIFIER_SYSTEM = "You are a fast classifier. Return ONLY strict JSON, no explanations."
CLASSIFIER_USER_TMPL = """User message:
{message}

Task: Detect if this contains a conditional instruction (if/when/unless … then …).
Output strict JSON with:
{
  "has_conditional": bool,
  "language": string,
  "predicate": {"normalized": string, "lhs": string|null, "op": string|null, "rhs": string|null,
               "needs_tools": [string], "confidence": number},
  "action": {"normalized": string, "verbs": [string], "objects": [string],
             "needs_tools": [string], "confidence": number},
  "tool_hints": [string],
  "confidence": number
}
Rules: concise; JSON only; no chain-of-thought."""

def _build_classifier_model() -> ChatGroq:
    from os import getenv
    api_key = getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing for classifier.")
    return ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.0, streaming=False)

def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _classify_conditional_sync(user_text: str) -> dict | None:
    """Synchronous, small, and strict JSON classification."""
    if not user_text:
        return None
    llm = _build_classifier_model()
    msg = CLASSIFIER_USER_TMPL.replace("{message}", user_text)
    resp = llm.invoke([SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=msg)])
    raw = (resp.content or "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None