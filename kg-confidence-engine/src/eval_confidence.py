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
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .access import ROLE_CLEARANCES, filtered_corpus  # noqa: E402
from .confidence import band_for, score_confidence  # noqa: E402
from .corpus import Corpus  # noqa: E402
from .decision import NODE_ID_RE, synthesize_decision  # noqa: E402
from .graph_builder import GraphBoostedRetriever, build_context_map  # noqa: E402
from .retrieval import build_retrievers  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_queries.json"
OUT_PATH = DATA_DIR / "confidence_eval.json"
CALIB_PNG = DATA_DIR / "calibration.png"
THESIS_PNG = DATA_DIR / "thesis_validation.png"
THESIS_JSON = DATA_DIR / "thesis_validation.json"


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


# ===========================================================================
# Phase 7 — correctness labeling, calibration + ECE, thesis validation
# ===========================================================================

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "into", "only", "are",
    "was", "were", "its", "it's", "during", "under", "which", "what", "when",
    "where", "there", "here", "than", "then", "also", "some", "such", "been",
    "have", "has", "had", "not", "but", "can", "could", "should", "would",
    "they", "their", "them", "your", "you", "via", "per", "all", "any", "more",
    "most", "other", "still", "just", "because", "about", "after", "before",
}


def significant_terms(text: str) -> set[str]:
    """Gold key terms: node IDs plus content words (>=4 chars, non-stopword)."""
    ids = {m.lower() for m in NODE_ID_RE.findall(text)}
    words = {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower()) if w not in _STOP}
    return ids | words


def judge_correct(qentry: dict, decision, *, tau: float = 0.35) -> tuple[bool, float]:
    """Keyed correctness: recall of gold key terms in the decision text.

    Role-adaptive by construction: surfacing the right (now-visible) nodes makes
    their gold content appear; when key evidence is hidden or missing, recall
    drops. An LLM-as-judge could replace this when ANTHROPIC_API_KEY is set.
    """
    gold_terms = significant_terms(qentry["gold_decision"])
    if not gold_terms:
        return False, 0.0
    text = decision.decision_text.lower()
    recall = sum(1 for t in gold_terms if t in text) / len(gold_terms)
    return recall >= tau, recall


def _struct_features(conf) -> dict:
    b = conf.breakdown
    return {
        "corroboration": b["corroboration"]["value"],
        "source_tier": b["source_tier"]["value"],
        "freshness": b["freshness"]["value"],
        "contradiction_free": 1.0 - b["contradiction"]["value"],
        "coverage_ok": 1.0 - b["coverage"]["value"],
        "sufficiency": conf.sufficiency,
    }


def generate_conditions(
    corpus: Corpus, queries: list[dict], *,
    roles=("intern", "engineer", "lead"), n_seeds_list=(6, 8, 10), tau: float = 0.35,
) -> list[dict]:
    """One row per (query, role, n_seeds): confidence, correctness, structure."""
    rows: list[dict] = []
    for role in roles:
        view = filtered_corpus(corpus, role)
        retriever = GraphBoostedRetriever(build_retrievers(view)["hybrid"], view)
        for q in queries:
            for n in n_seeds_list:
                top = [nid for nid, _ in retriever.retrieve(q["query"], k=n)]
                G = build_context_map(view, top, query=q["query"])
                decision = synthesize_decision(G, q["query"])
                conf = score_confidence(G, decision)
                correct, recall = judge_correct(q, decision, tau=tau)
                rows.append({
                    "id": q["id"], "role": role, "n_seeds": n,
                    "confidence": conf.score, "band": conf.band,
                    "correct": bool(correct), "gold_recall": round(recall, 3),
                    "nodes": G.number_of_nodes(),
                    **_struct_features(conf),
                })
    return rows


def calibration(rows: list[dict], n_bins: int = 5) -> tuple[list[dict], float]:
    """Bin by confidence; observed accuracy vs mean confidence; ECE."""
    conf = np.array([r["confidence"] for r in rows])
    correct = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(rows)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        c_mean = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        cnt = int(mask.sum())
        ece += (cnt / n) * abs(acc - c_mean)
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "count": cnt,
                     "mean_conf": round(c_mean, 3), "accuracy": round(acc, 3)})
    return bins, ece


def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def thesis_validation(rows: list[dict]) -> dict:
    """Does subgraph structure predict correctness? Correlations + grouped accuracy."""
    correct = [1.0 if r["correct"] else 0.0 for r in rows]
    feats = ["corroboration", "source_tier", "freshness", "contradiction_free",
             "coverage_ok", "sufficiency"]
    corrs = {f: round(_pearson([r[f] for r in rows], correct), 3) for f in feats}
    corrs["confidence"] = round(_pearson([r["confidence"] for r in rows], correct), 3)

    # The thesis is about DENSITY of corroboration. Per-node quality (freshness,
    # tier) is confounded by evidence volume: a thin subgraph can have high
    # per-node freshness yet be wrong for lack of corroboration. So split rich vs
    # poor on the density signal (corroboration + sufficiency), not on all six.
    density = np.array([np.mean([r["corroboration"], r["sufficiency"]]) for r in rows])
    med = float(np.median(density))
    rich = [correct[i] for i in range(len(rows)) if density[i] >= med]
    poor = [correct[i] for i in range(len(rows)) if density[i] < med]
    return {
        "n_conditions": len(rows),
        "overall_accuracy": round(float(np.mean(correct)), 3),
        "pearson_r_with_correctness": corrs,
        "split_on": "density (corroboration + sufficiency)",
        "rich_subgraph_accuracy": round(float(np.mean(rich)), 3) if rich else None,
        "poor_subgraph_accuracy": round(float(np.mean(poor)), 3) if poor else None,
        "rich_n": len(rich), "poor_n": len(poor),
    }


def _plot_reliability(bins: list[dict], ece: float) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    xs = [b["mean_conf"] for b in bins]
    ys = [b["accuracy"] for b in bins]
    sizes = [30 + 12 * b["count"] for b in bins]
    ax.plot(xs, ys, "-o", color="#1f77b4")
    ax.scatter(xs, ys, s=sizes, color="#1f77b4", alpha=0.4, zorder=3)
    ax.set_xlabel("predicted confidence (bin mean)")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CALIB_PNG, dpi=120); plt.close(fig)


def _plot_thesis(rows: list[dict], tv: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    # left: accuracy, rich vs poor subgraphs
    ax1.bar(["poor\nsubgraph", "rich\nsubgraph"],
            [tv["poor_subgraph_accuracy"], tv["rich_subgraph_accuracy"]],
            color=["#d62728", "#2ca02c"], alpha=0.8)
    ax1.set_ylim(0, 1); ax1.set_ylabel("decision accuracy")
    ax1.set_title("Accuracy by subgraph richness")
    for i, v in enumerate([tv["poor_subgraph_accuracy"], tv["rich_subgraph_accuracy"]]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center")
    # right: confidence distribution for correct vs incorrect
    corr = [r["confidence"] for r in rows if r["correct"]]
    inc = [r["confidence"] for r in rows if not r["correct"]]
    ax2.hist([inc, corr], bins=8, range=(0, 1), stacked=False,
             label=["incorrect", "correct"], color=["#d62728", "#2ca02c"], alpha=0.7)
    ax2.set_xlabel("confidence score"); ax2.set_ylabel("conditions")
    ax2.set_title("Confidence: correct vs incorrect"); ax2.legend()
    fig.tight_layout(); fig.savefig(THESIS_PNG, dpi=120); plt.close(fig)


def run_full(corpus: Corpus, queries: list[dict]) -> dict:
    rows = generate_conditions(corpus, queries)
    bins, ece = calibration(rows)
    tv = thesis_validation(rows)
    _plot_reliability(bins, ece)
    _plot_thesis(rows, tv)

    print("\n" + "=" * 64)
    print(f"CALIBRATION  ({len(rows)} conditions = "
          f"{len(queries)} queries x roles x n_seeds)")
    print("=" * 64)
    print(f"{'conf bin':<14} {'count':>6} {'mean_conf':>10} {'accuracy':>10}")
    print("-" * 64)
    for b in bins:
        print(f"[{b['lo']:.1f},{b['hi']:.1f}]   {b['count']:>6} "
              f"{b['mean_conf']:>10.3f} {b['accuracy']:>10.3f}")
    print("-" * 64)
    print(f"ECE = {ece:.3f}   overall accuracy = {tv['overall_accuracy']:.3f}")

    print("\n" + "=" * 64)
    print("THESIS VALIDATION — does subgraph structure predict correctness?")
    print("=" * 64)
    print("Pearson r with decision correctness:")
    for f, r in tv["pearson_r_with_correctness"].items():
        print(f"  {f:<20} r = {r:+.3f}")
    print(f"\naccuracy: rich subgraphs {tv['rich_subgraph_accuracy']:.3f} "
          f"(n={tv['rich_n']})  vs  poor subgraphs {tv['poor_subgraph_accuracy']:.3f} "
          f"(n={tv['poor_n']})")
    print("=" * 64)

    THESIS_JSON.write_text(json.dumps({"thesis": tv, "calibration": {"ece": ece, "bins": bins},
                                       "rows": rows}, indent=2))
    print(f"wrote {CALIB_PNG.name}, {THESIS_PNG.name}, {THESIS_JSON.name}")
    return {"ece": ece, "thesis": tv}


def main() -> int:
    full = len(sys.argv) > 1 and sys.argv[1] == "full"
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

    if full:
        run_full(corpus, queries)
    else:
        print("\n(run `python -m src.eval_confidence full` for calibration + "
              "thesis validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
