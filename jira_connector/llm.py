"""Ollama chat helper — turns JIRA search chunks into a grounded answer.

The connector itself is LLM-agnostic; this module is the optional CLI/RAG glue
that sends ACL-scoped, provenance-tagged JIRA chunks to a locally hosted Ollama
model and gets back a cited answer. Config via env:

    OLLAMA_HOST   (default http://localhost:11434)
    CHAT_MODEL    (default llama3)
"""

from __future__ import annotations

import os

import httpx

from .schema import Chunk

OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
CHAT_MODEL = os.environ.get("CHAT_MODEL") or "llama3"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT") or "120")

SYSTEM_PROMPT = (
    "You are an Escalation Context Assistant. Answer the user's question using "
    "ONLY the JIRA context provided. Rules:\n"
    "1. If the answer is not in the context, say so plainly — do not invent facts.\n"
    "2. Cite the ticket(s) you used inline using their key in brackets, e.g. [CXC-1234].\n"
    "3. Be concise and factual; focus on what happened, status, and next steps.\n"
)


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def build_context(chunks: list[Chunk]) -> str:
    """Render ranked chunks into a citable context block."""
    blocks = []
    for c in chunks:
        prov = c["provenance"]
        ticket = prov.get("ticket", "?")
        field = prov.get("field", "?")
        blocks.append(f"[{ticket}] (field: {field}) {prov.get('url','')}\n{c['text']}")
    return "\n\n".join(blocks)


def answer(question: str, chunks: list[Chunk], model: str | None = None) -> str:
    """Generate a grounded answer from JIRA chunks via Ollama."""
    if not chunks:
        return (
            "I couldn't find any in-scope JIRA tickets matching that question. "
            "Try broadening the query or adjusting the project/status scope."
        )

    context = build_context(chunks)
    user_prompt = (
        f"JIRA context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite ticket keys in brackets."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    with httpx.Client(base_url=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT) as client:
        try:
            resp = client.post(
                "/api/chat",
                json={"model": model or CHAT_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Failed to reach Ollama at {OLLAMA_HOST}: {exc}. "
                "Is `ollama serve` running and the model pulled?"
            ) from exc
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip() or "(empty response)"
