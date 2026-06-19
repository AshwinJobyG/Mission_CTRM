"""Vector RAG over the JIRA knowledge base.

This is the retrieval+generation half of the architecture flow:
    question -> ION embedding -> Chroma semantic search -> confidence
             -> grounded ION LLM answer with citations.

Confidence is derived from retrieval similarity (cosine), per the chosen design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import vectorstore
from .embeddings import embed_query
from .llm import run_chat
from .vectorstore import Retrieved

SYSTEM_PROMPT = (
    "You are an Escalation Context Assistant for JIRA. Answer the user's "
    "question using ONLY the provided context. Rules:\n"
    "1. If the answer is not in the context, say so plainly — do not invent facts.\n"
    "2. Cite the ticket(s) you used inline using their key in brackets, e.g. [NGPOWER-46].\n"
    "3. Be concise and factual; focus on status, what happened, and next steps.\n"
)


def _band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[Retrieved]
    confidence: float          # 0..1, overall
    used_context: bool = True
    per_source: list[tuple[str, float]] = field(default_factory=list)

    @property
    def confidence_pct(self) -> int:
        return round(self.confidence * 100)

    @property
    def confidence_band(self) -> str:
        return _band(self.confidence)


def _overall_confidence(sources: list[Retrieved]) -> float:
    """Blend the best match with the average of the top retrieved chunks."""
    if not sources:
        return 0.0
    sims = [s.similarity for s in sources]
    top = max(sims)
    avg = sum(sims) / len(sims)
    return round(0.6 * top + 0.4 * avg, 4)


def _build_context(sources: list[Retrieved]) -> str:
    blocks = []
    for s in sources:
        blocks.append(f"[{s.ticket}] (field: {s.field}, similarity {s.similarity:.2f}) {s.url}\n{s.text}")
    return "\n\n".join(blocks)


def answer(question: str, top_k: int = 6, model: str | None = None) -> RagAnswer:
    """Retrieve from the vector DB and generate a grounded, confidence-scored answer."""
    if vectorstore.count() == 0:
        return RagAnswer(
            question=question,
            answer=(
                "The JIRA vector index is empty. Build it first with "
                "`python -m jira_connector.cli index --project <KEY>`."
            ),
            sources=[],
            confidence=0.0,
            used_context=False,
        )

    qvec = embed_query(question)
    sources = vectorstore.query(qvec, top_k=top_k)
    confidence = _overall_confidence(sources)

    context = _build_context(sources)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite ticket keys in brackets."
    )
    reply = run_chat(SYSTEM_PROMPT, user_prompt, model=model)

    # Per-source confidence, de-duplicated by ticket (best chunk wins).
    best: dict[str, float] = {}
    for s in sources:
        best[s.ticket] = max(best.get(s.ticket, 0.0), s.similarity)
    per_source = sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    return RagAnswer(
        question=question,
        answer=reply,
        sources=sources,
        confidence=confidence,
        per_source=per_source,
    )
