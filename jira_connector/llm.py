"""Company-hosted LLM glue — turns JIRA search chunks into a grounded answer.

The connector itself is LLM-agnostic; this module is the optional CLI/RAG glue
that sends ACL-scoped, provenance-tagged JIRA chunks to the company-hosted,
OpenAI-compatible ION LLM (via langchain_openai) and gets back a cited answer.

Config via env (see .env.jira.example):
    ION_LLM_API_URL     base endpoint (the "/v1" suffix is added here)
    ION_LLM_API_KEY     bearer/API key
    ION_LLM_MODEL       model name
    ION_LLM_MAX_TOKENS  optional, default 30000

The langchain import is lazy so health/fetch/search work without it installed.
"""

from __future__ import annotations

import os

from .schema import Chunk

ION_LLM_API_KEY = os.environ.get("ION_LLM_API_KEY")
ION_LLM_API_URL = os.environ.get("ION_LLM_API_URL")
ION_LLM_MODEL = os.environ.get("ION_LLM_MODEL")
ION_LLM_MAX_TOKENS = int(os.environ.get("ION_LLM_MAX_TOKENS") or "30000")

SYSTEM_PROMPT = (
    "You are an Escalation Context Assistant. Answer the user's question using "
    "ONLY the JIRA context provided. Rules:\n"
    "1. If the answer is not in the context, say so plainly — do not invent facts.\n"
    "2. Cite the ticket(s) you used inline using their key in brackets, e.g. [CXC-1234].\n"
    "3. Be concise and factual; focus on what happened, status, and next steps.\n"
)


class LLMError(RuntimeError):
    """Raised when the ION LLM is misconfigured or unreachable."""


_llm = None


def _get_llm():
    """Build (once) a ChatOpenAI client pointed at the company-hosted endpoint."""
    global _llm
    if _llm is not None:
        return _llm

    if not (ION_LLM_API_URL and ION_LLM_MODEL):
        raise LLMError(
            "ION LLM not configured: set ION_LLM_API_URL and ION_LLM_MODEL "
            "(and ION_LLM_API_KEY) in the environment."
        )
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise LLMError(
            "langchain-openai is required for the ask/chat commands "
            "(pip install langchain-openai)."
        ) from exc

    # verify=False mirrors the company script (internal endpoint / custom CA).
    http_client = httpx.Client(verify=False)
    _llm = ChatOpenAI(
        base_url=f"{ION_LLM_API_URL}/v1",
        api_key=ION_LLM_API_KEY,
        model=ION_LLM_MODEL,
        max_tokens=ION_LLM_MAX_TOKENS,
        http_client=http_client,
    )
    return _llm


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
    """Generate a grounded answer from JIRA chunks via the ION LLM."""
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

    llm = _get_llm()
    if model:  # per-call model override
        llm = llm.bind(model=model)

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise LLMError(f"ION LLM request failed: {exc}") from exc

    content = getattr(response, "content", "") or ""
    return content.strip() or "(empty response)"
