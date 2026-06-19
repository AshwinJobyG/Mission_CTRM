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

    if record.get("summary"):
        chunks.append(_make(key, "summary", 0, record["summary"], url))

    for i, piece in enumerate(chunk_text(record.get("description") or "")):
        chunks.append(_make(key, "description", i, piece, url))

    for ci, comment in enumerate(record.get("comments") or []):
        body = comment.get("body") or ""
        for i, piece in enumerate(chunk_text(body)):
            chunks.append(_make(key, f"comment-{ci}", i, piece, url))

    return chunks
