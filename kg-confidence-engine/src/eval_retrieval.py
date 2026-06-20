"""Retrieval eval harness: P@k / R@k for keyword, embedding, and hybrid.

The methodology is part of the deliverable: we measure each retriever on the
labeled eval set rather than asserting that hybrid is best. Relevance is scored
against ``relevant_node_ids`` intersected with the corpus (a relevant id that is
intentionally absent — e.g. the dangling JIRA-4300 — cannot be retrieved and is
excluded from the denominator so it does not masquerade as a retrieval miss).

Run: ``python -m src.eval_retrieval [k]``  (default k=5)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .corpus import Corpus
from .retrieval import build_retrievers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_queries.json"
OUT_PATH = DATA_DIR / "retrieval_eval.json"


def load_queries() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["queries"]


def evaluate(corpus: Corpus, retrievers: dict, queries: list[dict], k: int = 5) -> dict:
    corpus_ids = corpus.ids
    results: dict[str, dict] = {}
    for key, retr in retrievers.items():
        p_sum = r_sum = 0.0
        n = 0
        per_query = []
        for q in queries:
            relevant = set(q["relevant_node_ids"]) & corpus_ids
            if not relevant:
                continue
            retrieved = [nid for nid, _ in retr.retrieve(q["query"], k=k)]
            hits = relevant & set(retrieved)
            p = len(hits) / k
            r = len(hits) / len(relevant)
            p_sum += p
            r_sum += r
            n += 1
            per_query.append(
                {"id": q["id"], "p_at_k": round(p, 3), "r_at_k": round(r, 3),
                 "hits": sorted(hits), "n_relevant": len(relevant)}
            )
        results[key] = {
            "name": retr.name,
            "p_at_k": round(p_sum / n, 4),
            "r_at_k": round(r_sum / n, 4),
            "per_query": per_query,
        }
    return results


def print_table(results: dict, k: int) -> None:
    print("=" * 64)
    print(f"RETRIEVAL EVAL — P@{k} / R@{k}  (mean over eval queries)")
    print("=" * 64)
    print(f"{'retriever':<28} {'P@'+str(k):>8} {'R@'+str(k):>8}")
    print("-" * 64)
    order = ["keyword", "embedding", "hybrid"]
    best = max(results.values(), key=lambda r: (r["p_at_k"], r["r_at_k"]))
    for key in order:
        r = results[key]
        flag = "  <- best P@k" if r is best else ""
        print(f"{r['name']:<28} {r['p_at_k']:>8.4f} {r['r_at_k']:>8.4f}{flag}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    k = int(argv[0]) if argv else 5
    corpus = Corpus.load()
    retrievers = build_retrievers(corpus)
    queries = load_queries()
    results = evaluate(corpus, retrievers, queries, k=k)
    print_table(results, k)
    summary = {
        "k": k,
        "embedding_backend": retrievers["embedding"].backend.name,
        "results": {key: {"name": r["name"], "p_at_k": r["p_at_k"], "r_at_k": r["r_at_k"]}
                    for key, r in results.items()},
        "per_query": {key: r["per_query"] for key, r in results.items()},
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
