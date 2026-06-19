"""Chroma vector store for JIRA knowledge.

We compute embeddings ourselves (via the ION endpoint) and pass them to Chroma,
keeping the store independent of any embedding backend. Cosine space is used so
similarity = 1 - distance gives a clean confidence score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunk import Chunk
from .embeddings import embed_texts

CHROMA_DIR = Path(os.environ.get("JIRA_CHROMA_DIR") or "./jira_chroma_db").expanduser()
COLLECTION = os.environ.get("JIRA_COLLECTION") or "jira_knowledge"


@dataclass
class Retrieved:
    text: str
    ticket: str
    field: str
    url: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [0, 1] (clamped) derived from cosine distance."""
        sim = 1.0 - self.distance
        return max(0.0, min(1.0, sim))


def _client():
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _collection():
    return _client().get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def reset() -> None:
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def count() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0


def add_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embed and upsert chunks. Returns the number stored."""
    if not chunks:
        return 0
    coll = _collection()
    stored = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([c["text"] for c in batch])
        coll.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "ticket": c["provenance"].get("ticket", "?"),
                    "field": c["provenance"].get("field", "?"),
                    "url": c["provenance"].get("url", ""),
                }
                for c in batch
            ],
        )
        stored += len(batch)
    return stored


def query(query_embedding: list[float], top_k: int = 6) -> list[Retrieved]:
    coll = _collection()
    if coll.count() == 0:
        return []
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out: list[Retrieved] = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        out.append(
            Retrieved(
                text=doc,
                ticket=meta.get("ticket", "?"),
                field=meta.get("field", "?"),
                url=meta.get("url", ""),
                distance=float(dist),
            )
        )
    return out
