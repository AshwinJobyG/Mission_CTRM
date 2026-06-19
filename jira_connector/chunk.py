"""Chunk long record fields into independently-citable pieces.

Each chunk carries its own provenance (which ticket, which field, the URL) so a
downstream answer can cite exactly where a statement came from.
"""

from __future__ import annotations

from .schema import Chunk, Record, chunk_provenance

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 120


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                pos = window.rfind(sep)
                if pos > size * 0.5:
                    end = start + pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _make(key: str, field: str, idx: int, text: str, url: str) -> Chunk:
    return Chunk(
        chunk_id=f"{key}#{field}#{idx}",
        text=text,
        score=0.0,  # filled in by rank.py
        provenance=chunk_provenance(ticket=key, field=field, url=url),
    )


def chunks_from_record(record: Record) -> list[Chunk]:
    """Produce provenance-tagged chunks for a normalized record."""
    key = record["id"]
    url = record["provenance"].get("url", "")
    chunks: list[Chunk] = []

    # Overview chunk: the structured fields (status, assignee, priority, dates,
    # labels, links) so questions like "what is the status of X" are retrievable.
    chunks.append(_make(key, "overview", 0, _overview_text(record), url))

    if record.get("summary"):
        chunks.append(_make(key, "summary", 0, record["summary"], url))

    for i, piece in enumerate(chunk_text(record.get("description") or "")):
        chunks.append(_make(key, "description", i, piece, url))

    for ci, comment in enumerate(record.get("comments") or []):
        body = comment.get("body") or ""
        for i, piece in enumerate(chunk_text(body)):
            chunks.append(_make(key, f"comment-{ci}", i, piece, url))

    return chunks


def _overview_text(record: Record) -> str:
    """A compact, human-readable summary of a ticket's structured fields."""

    def name(person):
        return person.get("name") if person else "Unassigned"

    links = ", ".join(f"{l.get('type')} {l.get('id')}" for l in record.get("links") or []) or "none"
    labels = ", ".join(record.get("labels") or []) or "none"
    lines = [
        f"Ticket {record.get('id')}",
        f"Summary: {record.get('summary') or '(none)'}",
        f"Status: {record.get('status') or 'unknown'}",
        f"Priority: {record.get('priority') or 'unknown'}",
        f"Assignee: {name(record.get('assignee'))}",
        f"Reporter: {name(record.get('reporter'))}",
        f"Created: {record.get('created') or 'unknown'}",
        f"Updated: {record.get('updated') or 'unknown'}",
        f"Labels: {labels}",
        f"Links: {links}",
    ]
    return "\n".join(lines)
