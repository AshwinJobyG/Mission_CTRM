"""Baseline relevance ranking: lexical overlap blended with recency.

This is the 24h baseline and a deliberate swap seam — a later layer can replace
``score_chunks`` with embedding/semantic re-ranking without touching search.py.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .jql import extract_terms
from .schema import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

LEXICAL_WEIGHT = 0.8
RECENCY_WEIGHT = 0.2


def _lexical_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    hits = sum(1 for term in terms if term.lower() in tokens)
    return hits / len(terms)


def _recency_score(updated: str | None) -> float:
    if not updated:
        return 0.0
    try:
        # JIRA timestamps look like 2026-04-22T14:03:00.000+0000
        cleaned = re.sub(r"\.\d+", "", updated).replace("Z", "+0000")
        dt = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S%z")
    except Exception:
        return 0.0
    days = max((datetime.now(timezone.utc) - dt).total_seconds() / 86400.0, 0.0)
    # Smooth decay: ~1.0 today, ~0.5 at 30 days, approaching 0 for old tickets.
    return 1.0 / (1.0 + math.log1p(days / 30.0))


# Chunks from a ticket the user named explicitly are pinned above keyword hits.
KEY_BOOST = 1.0


def score_chunks(
    chunks: list[Chunk],
    query: str,
    updated_by_ticket: dict[str, str] | None = None,
    boost_tickets: set[str] | None = None,
) -> list[Chunk]:
    """Assign a blended score to each chunk and return them sorted descending.

    ``boost_tickets`` are issue keys named explicitly in the query; their chunks
    get a large boost so they are guaranteed to lead the results.
    """
    terms = extract_terms(query)
    updated_by_ticket = updated_by_ticket or {}
    boost_tickets = boost_tickets or set()

    for chunk in chunks:
        ticket = chunk["provenance"].get("ticket", "")
        lexical = _lexical_score(chunk["text"], terms)
        recency = _recency_score(updated_by_ticket.get(ticket))
        score = LEXICAL_WEIGHT * lexical + RECENCY_WEIGHT * recency
        if ticket in boost_tickets:
            score += KEY_BOOST
        chunk["score"] = round(score, 4)

    return sorted(chunks, key=lambda c: c["score"], reverse=True)
