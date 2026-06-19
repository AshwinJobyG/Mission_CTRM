"""Local Chroma vector store wrapper.

We compute embeddings ourselves (via Ollama) and pass them directly to Chroma,
keeping the store decoupled from any embedding backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chunker import Chunk
from .config import CHROMA_DIR, COLLECTION, TOP_K
from .embeddings import embed_texts


@dataclass
class Retrieved:
    text: str
    source: str
    name: str
    path: str
    index: int
    distance: float


# Cache the Chroma collection at module scope. Building a PersistentClient
# reopens the on-disk index; doing it per query added seconds of overhead.
_COLLECTION = None


def _collection():
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    # Distance space cosine; we provide embeddings, so no embedding function.
    _COLLECTION = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _COLLECTION


def reset() -> None:
    """Delete and recreate the collection."""
    global _COLLECTION
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    _COLLECTION = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embed and upsert chunks. Returns the number of chunks stored."""
    if not chunks:
        return 0
    coll = _collection()
    stored = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([c.text for c in batch])
        coll.upsert(
            ids=[c.id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "source": c.source,
                    "name": c.name,
                    "path": c.path,
                    "index": c.index,
                }
                for c in batch
            ],
        )
        stored += len(batch)
    return stored


def count() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0


def query(text: str, top_k: int = TOP_K) -> list[Retrieved]:
    coll = _collection()
    n = coll.count()
    if n == 0:
        return []
    qvec = embed_texts([text])[0]
    res = coll.query(
        query_embeddings=[qvec],
        n_results=min(top_k, n),
        include=["documents", "metadatas", "distances"],
    )
    out: list[Retrieved] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        out.append(
            Retrieved(
                text=doc,
                source=meta.get("source", "?"),
                name=meta.get("name", "?"),
                path=meta.get("path", ""),
                index=int(meta.get("index", 0)),
                distance=float(dist),
            )
        )
    return out
