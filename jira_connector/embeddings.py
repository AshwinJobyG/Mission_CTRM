"""Pluggable embeddings for the JIRA vector DB.

Pick the backend with EMBED_PROVIDER (default "ion"):

  ion     -> company ION endpoint, OpenAI-compatible POST /v1/embeddings
             (ION_LLM_API_URL, ION_LLM_API_KEY, ION_LLM_EMBED_MODEL)
  ollama  -> local Ollama (OLLAMA_HOST, OLLAMA_EMBED_MODEL=nomic-embed-text)
  local   -> sentence-transformers (EMBED_LOCAL_MODEL=all-MiniLM-L6-v2)

If you are unsure whether ION serves embeddings, run:
    python -m jira_connector.cli embed-check
and switch EMBED_PROVIDER to whichever backend succeeds.
"""

from __future__ import annotations

import os
from typing import Sequence

EMBED_PROVIDER = (os.environ.get("EMBED_PROVIDER") or "ion").lower()

# ION
ION_LLM_API_KEY = os.environ.get("ION_LLM_API_KEY")
ION_LLM_API_URL = os.environ.get("ION_LLM_API_URL")
ION_LLM_EMBED_MODEL = os.environ.get("ION_LLM_EMBED_MODEL")
ION_VERIFY_SSL = (os.environ.get("ION_VERIFY_SSL") or "false").lower() not in {"false", "0", "no"}
ION_TIMEOUT = float(os.environ.get("ION_LLM_TIMEOUT") or os.environ.get("OLLAMA_TIMEOUT") or "120")

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL") or "nomic-embed-text"

# Local sentence-transformers
EMBED_LOCAL_MODEL = os.environ.get("EMBED_LOCAL_MODEL") or "all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Raised when the configured embeddings backend is misconfigured/unreachable."""


# --------------------------- ION ---------------------------
def _embed_ion(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    import httpx

    if not (ION_LLM_API_URL and ION_LLM_EMBED_MODEL):
        raise EmbeddingError(
            "ION embeddings not configured: set ION_LLM_API_URL and ION_LLM_EMBED_MODEL."
        )
    headers = {"Content-Type": "application/json"}
    if ION_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {ION_LLM_API_KEY}"

    vectors: list[list[float]] = []
    with httpx.Client(base_url=f"{ION_LLM_API_URL}/v1", timeout=ION_TIMEOUT, verify=ION_VERIFY_SSL) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                resp = client.post(
                    "/embeddings",
                    headers=headers,
                    json={"model": ION_LLM_EMBED_MODEL, "input": batch},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"ION embeddings request failed at {ION_LLM_API_URL}/v1/embeddings: {exc}. "
                    "If ION does not expose an embeddings model, set EMBED_PROVIDER=ollama."
                ) from exc
            data = resp.json().get("data", [])
            for item in sorted(data, key=lambda d: d.get("index", 0)):
                vectors.append(item["embedding"])
    return vectors


# --------------------------- Ollama ---------------------------
def _embed_ollama(texts: list[str]) -> list[list[float]]:
    import httpx

    with httpx.Client(base_url=OLLAMA_HOST, timeout=ION_TIMEOUT) as client:
        try:
            resp = client.post("/api/embed", json={"model": OLLAMA_EMBED_MODEL, "input": texts})
            if resp.status_code == 200:
                vecs = resp.json().get("embeddings")
                if vecs:
                    return vecs
        except httpx.HTTPError:
            pass
        # Fallback: one request per text.
        vectors: list[list[float]] = []
        for text in texts:
            try:
                resp = client.post("/api/embeddings", json={"model": OLLAMA_EMBED_MODEL, "prompt": text})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"Ollama embeddings failed at {OLLAMA_HOST}: {exc}. "
                    f"Is Ollama running and `ollama pull {OLLAMA_EMBED_MODEL}` done?"
                ) from exc
            vectors.append(resp.json()["embedding"])
        return vectors


# --------------------------- local ---------------------------
_local_model = None


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _local_model
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise EmbeddingError(
            "sentence-transformers not installed (pip install sentence-transformers)."
        ) from exc
    if _local_model is None:
        _local_model = SentenceTransformer(EMBED_LOCAL_MODEL)
    return _local_model.encode(list(texts), normalize_embeddings=True).tolist()


_PROVIDERS = {"ion": _embed_ion, "ollama": _embed_ollama, "local": _embed_local}


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Return one embedding per text using the configured backend."""
    if not texts:
        return []
    fn = _PROVIDERS.get(EMBED_PROVIDER)
    if fn is None:
        raise EmbeddingError(
            f"Unknown EMBED_PROVIDER '{EMBED_PROVIDER}'. Use one of: {', '.join(_PROVIDERS)}."
        )
    return fn(list(texts))


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def check() -> str:
    """Try a tiny embedding and report the provider + vector dimension (or error)."""
    vec = embed_texts(["connectivity check"])[0]
    return f"EMBED_PROVIDER={EMBED_PROVIDER} OK — embedding dimension {len(vec)}"
