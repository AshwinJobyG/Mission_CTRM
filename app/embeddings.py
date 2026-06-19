"""Embeddings via the local Ollama server."""

from __future__ import annotations

from typing import Sequence

import httpx

from .config import EMBED_MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


def _client() -> httpx.Client:
    return httpx.Client(base_url=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)


def embed_texts(texts: Sequence[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Return one embedding vector per input text.

    Uses Ollama's batch ``/api/embed`` endpoint when available and falls back
    to the older single-input ``/api/embeddings`` endpoint.
    """
    if not texts:
        return []

    with _client() as client:
        # Preferred: batch endpoint (Ollama >= 0.1.39).
        try:
            resp = client.post("/api/embed", json={"model": model, "input": list(texts)})
            if resp.status_code == 200:
                data = resp.json()
                vectors = data.get("embeddings")
                if vectors:
                    return vectors
        except httpx.HTTPError:
            pass

        # Fallback: one request per text.
        vectors = []
        for text in texts:
            try:
                resp = client.post("/api/embeddings", json={"model": model, "prompt": text})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(
                    f"Failed to get embeddings from Ollama at {OLLAMA_HOST}: {exc}"
                ) from exc
            vec = resp.json().get("embedding")
            if not vec:
                raise OllamaError("Ollama returned an empty embedding.")
            vectors.append(vec)
        return vectors


def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    return embed_texts([text], model=model)[0]
