"""Confidence eval harness.

Part 1 (this phase): run the full pipeline over every eval query and check that
the predicted confidence lands in the expected band (high/medium/low) for the
labeled queries.

Part 2 (Phase 7): correctness labeling, calibration curve + ECE, and the thesis
validation experiment (does subgraph structure predict correctness?).

Run: ``python -m src.eval_confidence``
"""

from __future__ import annotations

import json
from pathlib import Path

from .confidence import band_for, score_confidence
from .corpus import Corpus
from .decision import synthesize_decision
from .graph_builder import GraphBoostedRetriever, build_context_map
from .retrieval import build_retrievers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_queries.json"
OUT_PATH = DATA_DIR / "confidence_eval.json"


def load_queries() -> list[dict]:
    return json.loads(EVAL_PATH.read_text())["queries"]


def run_pipeline(corpus, retriever, query: str, *, n_seeds: int = 8):
    top = [nid for nid, _ in retriever.retrieve(query, k=n_seeds)]
    G = build_context_map(corpus, top, query=query)
    decision = synthesize_decision(G, query)
    conf = score_confidence(G, decision)
    return G, decision, conf


def evaluate_bands(corpus: Corpus, queries: list[dict], *, n_seeds: int = 8) -> list[dict]:
    retriever = GraphBoostedRetriever(build_retrievers(corpus)["hybrid"], corpus)
    rows = []
    for q in queries:
        _, decision, conf = run_pipeline(corpus, retriever, q["query"], n_seeds=n_seeds)
        rows.append({
            "id": q["id"],
            "expected": q["expected_confidence_band"],
            "score": conf.score,
            "predicted": conf.band,
            "match": conf.band == q["expected_confidence_band"],
            "method": decision.method,
        })
    return rows


def main() -> int:
    corpus = Corpus.load()
    queries = load_queries()
    rows = evaluate_bands(corpus, queries)

    print("=" * 64)
    print("CONFIDENCE — predicted band vs expected band")
    print("=" * 64)
    print(f"{'query':<6} {'score':>7} {'predicted':>10} {'expected':>10}  {'':>3}")
    print("-" * 64)
    for r in rows:
        flag = "ok" if r["match"] else "XX"
        print(f"{r['id']:<6} {r['score']:>7.3f} {r['predicted']:>10} {r['expected']:>10}  {flag:>3}")
    n_match = sum(r["match"] for r in rows)
    print("-" * 64)
    print(f"band-match accuracy: {n_match}/{len(rows)} = {n_match/len(rows):.2f}")
    print("=" * 64)

    # score ranges per expected band (useful for setting/validating thresholds)
    by_band: dict[str, list[float]] = {}
    for r in rows:
        by_band.setdefault(r["expected"], []).append(r["score"])
    print("\nscore range by EXPECTED band:")
    for band in ("high", "medium", "low"):
        xs = sorted(by_band.get(band, []))
        if xs:
            print(f"  {band:<7} n={len(xs)}  min={xs[0]:.3f}  max={xs[-1]:.3f}  "
                  f"mean={sum(xs)/len(xs):.3f}")

    OUT_PATH.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
