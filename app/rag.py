"""Retrieval-augmented generation against the local Ollama LLM.

The prompt forces the model to answer *only* from the retrieved context, cite
sources, and admit when the answer is not present — matching the PS-019
constraints (provenance, no hallucination).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx

from . import vectorstore
from .config import (
    CHAT_MODEL,
    EMBED_MODEL,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    TOP_K,
)
from .embeddings import OllamaError, _client, embed_texts
from .vectorstore import Retrieved

SYSTEM_PROMPT = (
    "You are an Enterprise Knowledge Retention & Discovery Assistant. "
    "Answer the user's question using ONLY the information in the provided "
    "context. Follow these rules strictly:\n"
    "1. If the answer is not in the context, say you don't know and suggest "
    "what source or SME validation is needed. Do not invent facts.\n"
    "2. Cite the sources you used inline using their [number] markers.\n"
    "3. Be concise and factual. Prefer reusable, source-backed answers.\n"
)

# Per-request tuning sent to Ollama. keep_alive keeps the model resident so we
# don't pay the load cost on every query; the options bound generation work.
_OPTIONS = {"num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Retrieved]
    used_context: bool

    def format_sources(self) -> str:
        if not self.sources:
            return "(no sources)"
        seen: dict[str, int] = {}
        lines = []
        for r in self.sources:
            if r.source in seen:
                continue
            seen[r.source] = len(seen) + 1
            lines.append(f"  [{seen[r.source]}] {r.source}")
        return "\n".join(lines)


@dataclass
class StreamingAnswer:
    """Sources are known up front; ``tokens`` yields the answer incrementally."""

    question: str
    sources: list[Retrieved]
    used_context: bool
    tokens: Iterator[str]


def _build_context(chunks: list[Retrieved]) -> tuple[str, dict[str, int]]:
    """Return a numbered context block and a source->number map."""
    numbering: dict[str, int] = {}
    blocks = []
    for chunk in chunks:
        if chunk.source not in numbering:
            numbering[chunk.source] = len(numbering) + 1
        n = numbering[chunk.source]
        blocks.append(f"[{n}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n".join(blocks), numbering


def _payload(messages: list[dict], model: str, *, stream: bool) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": _OPTIONS,
    }


def _chat(messages: list[dict], model: str) -> str:
    client = _client()
    try:
        resp = client.post("/api/chat", json=_payload(messages, model, stream=False))
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Failed to reach Ollama chat API at {OLLAMA_HOST}: {exc}. "
            "If this is a timeout, the model is likely cold-starting — make sure "
            f"`ollama serve` is running and `{CHAT_MODEL}` is pulled, or raise "
            "OLLAMA_TIMEOUT / use a smaller CHAT_MODEL."
        ) from exc
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip()


def _chat_stream(messages: list[dict], model: str) -> Iterator[str]:
    """Yield answer fragments as Ollama produces them (lower time-to-first-token)."""
    client = _client()
    try:
        with client.stream(
            "POST", "/api/chat", json=_payload(messages, model, stream=True)
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
            f"Failed to reach Ollama chat API at {OLLAMA_HOST}: {exc}. "
            "If this is a timeout, the model is likely cold-starting — make sure "
            f"`ollama serve` is running and `{CHAT_MODEL}` is pulled, or raise "
            "OLLAMA_TIMEOUT / use a smaller CHAT_MODEL."
        ) from exc


def _no_context_message() -> str:
    return (
        "I don't have any indexed knowledge to answer this yet. "
        "Ingest a folder first (see the `ingest` command), or the "
        "question may fall outside the available sources."
    )


def _messages(question: str, chunks: list[Retrieved]) -> list[dict]:
    context, _ = _build_context(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite sources with [number]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def answer_question(
    question: str,
    top_k: int = TOP_K,
    model: str = CHAT_MODEL,
) -> Answer:
    """Retrieve context for ``question`` and generate a grounded answer."""
    chunks = vectorstore.query(question, top_k=top_k)

    if not chunks:
        return Answer(
            question=question,
            answer=_no_context_message(),
            sources=[],
            used_context=False,
        )

    reply = _chat(_messages(question, chunks), model=model)
    return Answer(
        question=question,
        answer=reply or "(the model returned an empty response)",
        sources=chunks,
        used_context=True,
    )


def stream_question(
    question: str,
    top_k: int = TOP_K,
    model: str = CHAT_MODEL,
) -> StreamingAnswer:
    """Like :func:`answer_question`, but streams the answer token-by-token."""
    chunks = vectorstore.query(question, top_k=top_k)

    if not chunks:
        return StreamingAnswer(
            question=question,
            sources=[],
            used_context=False,
            tokens=iter([_no_context_message()]),
        )

    return StreamingAnswer(
        question=question,
        sources=chunks,
        used_context=True,
        tokens=_chat_stream(_messages(question, chunks), model=model),
    )


def warm_up(model: str = CHAT_MODEL) -> None:
    """Pre-load the chat and embed models so the first query isn't a cold start.

    Loading both keeps them resident together, avoiding the chat<->embed model
    swap that otherwise reloads a model on every query. Failures are ignored —
    warm-up is best-effort.
    """
    try:
        embed_texts([" "], model=EMBED_MODEL)
    except OllamaError:
        pass
    try:
        client = _client()
        client.post(
            "/api/chat",
            json=_payload(
                [{"role": "user", "content": "ok"}], model, stream=False
            ),
        )
    except httpx.HTTPError:
        pass
