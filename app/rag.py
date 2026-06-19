"""Retrieval-augmented generation against the local Ollama LLM.

The prompt forces the model to answer *only* from the retrieved context, cite
sources, and admit when the answer is not present — matching the PS-019
constraints (provenance, no hallucination).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from . import vectorstore
from .config import CHAT_MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT, TOP_K
from .embeddings import OllamaError
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


def _chat(messages: list[dict], model: str) -> str:
    with httpx.Client(base_url=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT) as client:
        try:
            resp = client.post(
                "/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Failed to reach Ollama chat API at {OLLAMA_HOST}: {exc}"
            ) from exc
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip()


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
            answer=(
                "I don't have any indexed knowledge to answer this yet. "
                "Ingest a folder first (see the `ingest` command), or the "
                "question may fall outside the available sources."
            ),
            sources=[],
            used_context=False,
        )

    context, _ = _build_context(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite sources with [number]."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    reply = _chat(messages, model=model)
    return Answer(
        question=question,
        answer=reply or "(the model returned an empty response)",
        sources=chunks,
        used_context=True,
    )
