"""Split document text into overlapping chunks for embedding."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .loader import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str   # relative file path
    name: str     # file name
    path: str     # absolute path
    index: int    # chunk index within the document


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        return [text]
    if overlap >= size:
        overlap = size // 4

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        # Prefer to break on a paragraph/sentence boundary near the end.
        if end < length:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                pos = window.rfind(sep)
                if pos > size * 0.5:
                    end = start + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(
    docs: list[Document],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(_split_text(doc.text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{doc.source}::chunk-{i}",
                    text=piece,
                    source=doc.source,
                    name=doc.name,
                    path=doc.path,
                    index=i,
                )
            )
    return chunks
