"""Embeddings via the company-hosted ION LLM (OpenAI-compatible endpoint).

Calls POST {ION_LLM_API_URL}/v1/embeddings directly with httpx so we control
batching, timeouts and TLS (verify=False mirrors the ION chat client). Config:

    ION_LLM_API_URL        base endpoint (the "/v1" suffix is added here)
    ION_LLM_API_KEY        bearer/API key
    ION_LLM_EMBED_MODEL    embeddings model name (REQUIRED for the vector DB)
    ION_VERIFY_SSL         "true" to verify TLS (default false, like the chat client)
"""

from __future__ import annotations

import os
from typing import Sequence

import httpx

ION_LLM_API_KEY = os.environ.get("ION_LLM_API_KEY")
ION_LLM_API_URL = os.environ.get("ION_LLM_API_URL")
ION_LLM_EMBED_MODEL = os.environ.get("ION_LLM_EMBED_MODEL")
ION_VERIFY_SSL = (os.environ.get("ION_VERIFY_SSL") or "false").lower() not in {"false", "0", "no"}
ION_TIMEOUT = float(os.environ.get("ION_LLM_TIMEOUT") or os.environ.get("OLLAMA_TIMEOUT") or "120")


class EmbeddingError(RuntimeError):
    """Raised when the ION embeddings endpoint is misconfigured or unreachable."""


def _check_config() -> None:
    if not (ION_LLM_API_URL and ION_LLM_EMBED_MODEL):
        raise EmbeddingError(
            "ION embeddings not configured: set ION_LLM_API_URL and "
            "ION_LLM_EMBED_MODEL (and ION_LLM_API_KEY) in the environment."
        )


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
    """Return one embedding vector per input text (batched)."""
    _check_config()
    if not texts:
        return []

    headers = {"Content-Type": "application/json"}
    if ION_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {ION_LLM_API_KEY}"

    vectors: list[list[float]] = []
    with httpx.Client(base_url=f"{ION_LLM_API_URL}/v1", timeout=ION_TIMEOUT, verify=ION_VERIFY_SSL) as client:
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            try:
                resp = client.post(
                    "/embeddings",
                    headers=headers,
                    json={"model": ION_LLM_EMBED_MODEL, "input": batch},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"ION embeddings request failed at {ION_LLM_API_URL}/v1/embeddings: {exc}"
                ) from exc
            data = resp.json().get("data", [])
            # Preserve input order (OpenAI returns an 'index' per item).
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            for item in ordered:
                vectors.append(item["embedding"])
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
