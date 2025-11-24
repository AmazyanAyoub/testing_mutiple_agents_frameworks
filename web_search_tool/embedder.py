import httpx
from typing import Any, Dict, List, Union

class OllamaEmbedder:
    """
    Minimal embedder that calls an Ollama server for embeddings.
    Expects the server to support POST /api/embeddings with JSON {model, prompt}.
    """

    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self._http = httpx.AsyncClient()

    async def embed(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        is_list = isinstance(texts, list)
        batch = texts if is_list else [texts]

        embs: List[Any] = []
        for t in batch:
            resp = await self._http.post(
                f"{self.endpoint}/api/embeddings",
                json={"model": self.model_name, "prompt": t},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding") or data.get("embeddings")
            if emb is None:
                raise ValueError("No embedding field in Ollama response")
            embs.append(emb)

        return {"embeddings": embs if is_list else embs[0]}
