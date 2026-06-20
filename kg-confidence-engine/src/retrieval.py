"""Retrieval layer: keyword (BM25), embedding (dense), and hybrid (RRF).

The default retriever is hybrid: keyword and embedding rankings fused by
Reciprocal Rank Fusion. We keep the two stages distinct (the GBrain
``search`` discipline) and measure each one (see ``eval_retrieval.py``) so the
choice of hybrid is justified by P@5/R@5, not asserted.

Embedding backend
-----------------
The intended embedder is ``sentence-transformers/all-MiniLM-L6-v2``. It is used
automatically whenever the weights are loadable (HuggingFace egress, or a local
model directory via ``KGCE_ST_MODEL``). When the model cannot be loaded (e.g. a
sandbox with HF blocked) we fall back to a deterministic, network-free dense
embedder (hashed char+word n-gram TF-IDF). The retriever interface and the RRF
fusion are identical either way, so swapping in MiniLM is a zero-code change.
Set ``KGCE_EMBED_BACKEND=hashing|st|auto`` to force a backend (default ``auto``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Corpus

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def node_text(node: dict) -> str:
    """The text the retrievers search over: title + body."""
    return f"{node['title']} {node['body']}"


# ============================================================================
# Embedding backends
# ============================================================================

class EmbeddingBackend(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingTfidfBackend:
    """Deterministic, network-free dense embedder.

    Hashes word unigrams and character n-grams into a fixed-width vector,
    weighted by sublinear TF and corpus IDF, then L2-normalizes. Cosine
    similarity over these vectors is a genuine lexical-semantic signal that is
    meaningfully distinct from BM25's exact word matching (it is robust to
    morphology and vocabulary mismatch via subword n-grams).
    """

    name = "hashing-tfidf"

    def __init__(self, dim: int = 4096, char_ngrams: tuple[int, ...] = (3, 4, 5)):
        self.dim = dim
        self.char_ngrams = char_ngrams
        self._idf: dict[str, float] = {}
        self._default_idf: float = 1.0

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        feats: list[str] = [f"w:{t}" for t in tokenize(text)]
        compact = re.sub(r"\s+", " ", text)
        for n in self.char_ngrams:
            for i in range(len(compact) - n + 1):
                feats.append(f"c{n}:{compact[i:i + n]}")
        return feats

    def fit(self, texts: Sequence[str]) -> None:
        n_docs = len(texts)
        df: dict[str, int] = {}
        for t in texts:
            for f in set(self._features(t)):
                df[f] = df.get(f, 0) + 1
        self._idf = {f: math.log((n_docs + 1) / (d + 1)) + 1.0 for f, d in df.items()}
        # Unseen features at query time get the highest (rarest) IDF.
        self._default_idf = (max(self._idf.values()) if self._idf else 1.0)

    def _hash(self, feature: str) -> tuple[int, float]:
        h = hashlib.md5(feature.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % self.dim
        sign = 1.0 if (h[4] & 1) else -1.0
        return idx, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            counts: dict[str, int] = {}
            for f in self._features(t):
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                idf = self._idf.get(f, self._default_idf)
                weight = (1.0 + math.log(c)) * idf
                idx, sign = self._hash(f)
                out[r, idx] += sign * weight
            norm = np.linalg.norm(out[r])
            if norm > 0:
                out[r] /= norm
        return out


class SentenceTransformerBackend:
    """Wraps sentence-transformers/all-MiniLM-L6-v2 (the intended embedder)."""

    name = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # local import

        model_name = model_name or os.environ.get("KGCE_ST_MODEL", "all-MiniLM-L6-v2")
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> None:  # no corpus fitting needed
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def get_embedding_backend(prefer: str | None = None) -> EmbeddingBackend:
    """Pick an embedding backend. ``auto`` tries MiniLM, falls back to hashing."""
    prefer = (prefer or os.environ.get("KGCE_EMBED_BACKEND", "auto")).lower()
    if prefer in ("st", "sentence-transformers", "minilm", "auto"):
        try:
            return SentenceTransformerBackend()
        except Exception as exc:  # offline / model unavailable
            if prefer != "auto":
                raise
            print(
                f"[retrieval] all-MiniLM-L6-v2 unavailable ({type(exc).__name__}); "
                "falling back to deterministic hashing-tfidf embedder. "
                "Enable HuggingFace egress or set KGCE_ST_MODEL to use MiniLM.",
                file=sys.stderr,
            )
    return HashingTfidfBackend()


# ============================================================================
# Retrievers — common interface: retrieve(query, k) -> list[(node_id, score)]
# ============================================================================

class KeywordRetriever:
    name = "keyword (BM25)"

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.ids = list(corpus.order)
        self._bm25 = BM25Okapi([tokenize(node_text(corpus.nodes[i])) for i in self.ids])

    def scores(self, query: str) -> dict[str, float]:
        raw = self._bm25.get_scores(tokenize(query))
        return {nid: float(s) for nid, s in zip(self.ids, raw)}

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        return _topk(self.scores(query), k)


class EmbeddingRetriever:
    def __init__(self, corpus: Corpus, backend: EmbeddingBackend | None = None):
        self.corpus = corpus
        self.ids = list(corpus.order)
        self.backend = backend or get_embedding_backend()
        self.name = f"embedding ({self.backend.name})"
        texts = [node_text(corpus.nodes[i]) for i in self.ids]
        self.backend.fit(texts)
        self.matrix = self._load_or_compute(texts)

    def _cache_key(self, texts: Sequence[str]) -> str:
        h = hashlib.md5()
        h.update(self.backend.name.encode())
        for nid, t in zip(self.ids, texts):
            h.update(nid.encode())
            h.update(t.encode("utf-8"))
        return h.hexdigest()[:16]

    def _load_or_compute(self, texts: Sequence[str]) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(texts)
        npy = CACHE_DIR / f"emb_{key}.npy"
        if npy.exists():
            return np.load(npy)
        matrix = self.backend.encode(texts)
        np.save(npy, matrix)
        return matrix

    def scores(self, query: str) -> dict[str, float]:
        q = self.backend.encode([query])[0]
        sims = self.matrix @ q  # rows are L2-normalized => cosine
        return {nid: float(s) for nid, s in zip(self.ids, sims)}

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        return _topk(self.scores(query), k)


class HybridRetriever:
    """Reciprocal Rank Fusion over keyword and embedding rankings."""

    name = "hybrid (RRF)"

    def __init__(
        self,
        keyword: KeywordRetriever,
        embedding: EmbeddingRetriever,
        k_rrf: int = 60,
    ):
        self.keyword = keyword
        self.embedding = embedding
        self.k_rrf = k_rrf

    def scores(self, query: str) -> dict[str, float]:
        fused: dict[str, float] = {}
        for retr in (self.keyword, self.embedding):
            ranked = _rank_ids(retr.scores(query))
            for rank, nid in enumerate(ranked, start=1):
                fused[nid] = fused.get(nid, 0.0) + 1.0 / (self.k_rrf + rank)
        return fused

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        return _topk(self.scores(query), k)


# ---- helpers ----------------------------------------------------------------

def _rank_ids(scores: dict[str, float]) -> list[str]:
    return [nid for nid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _topk(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]


def build_retrievers(
    corpus: Corpus, backend: EmbeddingBackend | None = None
) -> dict[str, object]:
    """Construct the three retrievers sharing one embedding backend."""
    kw = KeywordRetriever(corpus)
    emb = EmbeddingRetriever(corpus, backend)
    hyb = HybridRetriever(kw, emb)
    return {"keyword": kw, "embedding": emb, "hybrid": hyb}


if __name__ == "__main__":
    corpus = Corpus.load()
    retrievers = build_retrievers(corpus)
    demo = "what is the root cause of the SG settlement batch failures?"
    print(f"query: {demo}\n")
    for key, r in retrievers.items():
        print(f"--- {r.name} ---")
        for nid, score in r.retrieve(demo, k=5):
            print(f"  {score:7.4f}  {nid:10} {corpus.nodes[nid]['title']}")
        print()
