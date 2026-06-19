"""Ollama chat helper — turns JIRA search chunks into a grounded answer.

The connector itself is LLM-agnostic; this module is the optional CLI/RAG glue
that sends ACL-scoped, provenance-tagged JIRA chunks to a locally hosted Ollama
model and gets back a cited answer. Config via env:

    OLLAMA_HOST        (default http://localhost:11434)
    CHAT_MODEL         (default llama3)
    OLLAMA_TIMEOUT     (default 300 seconds — read budget for load+generation)
    OLLAMA_KEEP_ALIVE  (default 30m — keeps the model resident between queries)
    NUM_PREDICT        (default 512 — cap on generated tokens)
    NUM_CTX            (default 4096 — context window)
"""

from __future__ import annotations

import atexit
import json
import os
from typing import Iterator

import httpx

from .schema import Chunk

OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
CHAT_MODEL = os.environ.get("CHAT_MODEL") or "llama3"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT") or "300")
OLLAMA_CONNECT_TIMEOUT = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT") or "10")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE") or "30m"
NUM_PREDICT = int(os.environ.get("NUM_PREDICT") or "512")
NUM_CTX = int(os.environ.get("NUM_CTX") or "4096")

_OPTIONS = {"num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}

SYSTEM_PROMPT = (
    "You are an Escalation Context Assistant. Answer the user's question using "
    "ONLY the JIRA context provided. Rules:\n"
    "1. If the answer is not in the context, say so plainly — do not invent facts.\n"
    "2. Cite the ticket(s) you used inline using their key in brackets, e.g. [CXC-1234].\n"
    "3. Be concise and factual; focus on what happened, status, and next steps.\n"
)


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


# Reuse one client across calls to avoid per-request connection setup.
_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        timeout = httpx.Timeout(OLLAMA_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT)
        _CLIENT = httpx.Client(base_url=OLLAMA_HOST, timeout=timeout)
        atexit.register(_CLIENT.close)
    return _CLIENT


def build_context(chunks: list[Chunk]) -> str:
    """Render ranked chunks into a citable context block."""
    blocks = []
    for c in chunks:
        prov = c["provenance"]
        ticket = prov.get("ticket", "?")
        field = prov.get("field", "?")
        blocks.append(f"[{ticket}] (field: {field}) {prov.get('url','')}\n{c['text']}")
    return "\n\n".join(blocks)


def _messages(question: str, chunks: list[Chunk]) -> list[dict]:
    context = build_context(chunks)
    user_prompt = (
        f"JIRA context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite ticket keys in brackets."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _payload(messages: list[dict], model: str, *, stream: bool) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": _OPTIONS,
    }


_NO_CONTEXT = (
    "I couldn't find any in-scope JIRA tickets matching that question. "
    "Try broadening the query or adjusting the project/status scope."
)


def answer(question: str, chunks: list[Chunk], model: str | None = None) -> str:
    """Generate a grounded answer from JIRA chunks via Ollama."""
    if not chunks:
        return _NO_CONTEXT

    try:
        resp = _client().post(
            "/api/chat", json=_payload(_messages(question, chunks), model or CHAT_MODEL, stream=False)
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Failed to reach Ollama at {OLLAMA_HOST}: {exc}. "
            "Is `ollama serve` running and the model pulled?"
        ) from exc
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip() or "(empty response)"


def answer_stream(
    question: str, chunks: list[Chunk], model: str | None = None
) -> Iterator[str]:
    """Stream a grounded answer fragment-by-fragment (lower time-to-first-token)."""
    if not chunks:
        yield _NO_CONTEXT
        return

    try:
        with _client().stream(
            "POST",
            "/api/chat",
            json=_payload(_messages(question, chunks), model or CHAT_MODEL, stream=True),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    yield piece
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Failed to reach Ollama at {OLLAMA_HOST}: {exc}. "
            "Is `ollama serve` running and the model pulled?"
        ) from exc
