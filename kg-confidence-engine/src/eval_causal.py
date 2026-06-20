"""Phase E — causal eval: path-retrieval accuracy, calibration, thesis (causal).

Measures the new causal capability with the same rigor as the factual set, in
**new files only** (the original eval set and its numbers are untouched):

1. **Path-retrieval accuracy** — did Stage 3.5 recover the gold causal path?
   Precision / recall over path edges (src, rel, target) vs the manifest gold.
2. **Calibration (causal)** — feeds causal ``(predicted_confidence, correct)``
   pairs into the *existing* calibration harness (``eval_confidence.calibration``)
   so the reliability curve / ECE now also cover causal questions.
3. **Thesis (causal)** — higher ``path_completeness`` correlates with decision
   correctness, demonstrated on a controlled evidence-degradation sweep.

Run: ``python -m src.eval_causal``  (writes data/causal_eval.json [+ png])
"""

from __future__ import annotations

import json
from pathlib import Path

from .causal_traversal import (CausalPath, CausalStep, answer_query,
                               classify_query, trace_causal_path)
from .confidence import score_confidence
from .corpus import Corpus
from .decision import synthesize_decision
from .eval_confidence import _pearson, calibration, judge_correct
from .graph_builder import build_context_map

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_causal.json"
OUT_PATH = DATA_DIR / "causal_eval.json"
CALIB_PNG = DATA_DIR / "causal_calibration.png"

# Stricter gold-recovery bar for causal "why" answers (a fragment of the chain
# is not a correct causal explanation). Local to this harness.
CAUSAL_TAU = 0.7


def load_causal_eval() -> dict:
    return json.loads(EVAL_PATH.read_text())


# ---------------------------------------------------------------------------
# 1. Path-retrieval accuracy
# ---------------------------------------------------------------------------

def _edge_set(edges) -> set[tuple]:
    return {tuple(e) for e in edges}


def path_retrieval_scores(corpus: Corpus, q: dict) -> dict:
    """Precision / recall of the traced path edges against the gold edges."""
    path = trace_causal_path(corpus, q["query"], entry=q.get("entry_node"))
    pred = _edge_set(path.edges())
    gold = _edge_set(q["gold_path_edges"])
    hits = pred & gold
    precision = len(hits) / len(pred) if pred else 0.0
    recall = len(hits) / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    root_ok = q.get("gold_root_cause") in path.root_causes
    personas_ok = all(path.personas.get(role) == nid
                      for role, nid in q.get("gold_personas", {}).items()
                      if role in path.personas)
    return {
        "id": q["id"], "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3), "n_pred": len(pred), "n_gold": len(gold),
        "root_cause_recovered": bool(root_ok), "personas_consistent": bool(personas_ok),
        "missing_edges": sorted(str(e) for e in (gold - pred)),
    }


# ---------------------------------------------------------------------------
# 2. Causal conditions → calibration + (3) thesis
# ---------------------------------------------------------------------------

_PERSONA_BY_REL = {"PRIORITIZED_BY": "made_the_call", "OWNED_BY": "owned_the_code"}


def _prefix(full: CausalPath, k: int) -> CausalPath:
    """A structurally-truncated path: the first ``k`` steps, with root causes and
    personas recomputed from what remains.

    Truncation is the honest degradation for the thesis: a shorter chain reaches
    fewer causes and names fewer people, so BOTH path_completeness AND the gold
    content recoverable from the narration fall together — no circularity.
    """
    steps = full.steps[:k]
    reached = {full.entry} | {s.src for s in steps} | {s.target for s in steps}
    out_src = {s.src for s in steps}
    type_of = {s.target: s.target_type for s in steps}
    root = sorted(nid for nid in reached
                  if type_of.get(nid) == "risk"
                  or (type_of.get(nid) == "ticket" and nid not in out_src))
    personas: dict[str, str] = {}
    for s in steps:
        if s.rel in _PERSONA_BY_REL and s.target_type == "person":
            personas.setdefault(_PERSONA_BY_REL[s.rel], s.target)
        elif s.rel == "RAISED_RISK" and s.target_type == "person":
            personas.setdefault("raised_risk", s.target)
    return CausalPath(entry=full.entry, steps=steps, root_causes=root, personas=personas)


def generate_causal_conditions(corpus: Corpus, queries: list[dict]) -> list[dict]:
    """One row per (causal query × path-prefix length k).

    Sweeping k from a 2-step stub up to the full chain varies how complete the
    traced/narrated path is. The context map is rebuilt from only the retained
    path nodes, so the decision can recall only the gold content those nodes
    carry — giving the spread needed to test the thesis on the causal set.
    """
    rows: list[dict] = []
    for q in queries:
        if q.get("expected_type") != "causal":
            continue
        full = trace_causal_path(corpus, q["query"], entry=q.get("entry_node"))
        n = len(full.steps)
        for k in range(2, n + 1):
            path = _prefix(full, k)
            seeds = [nid for nid in path.context_ids() if nid in corpus]
            G = build_context_map(corpus, seeds, query=q["query"])
            decision = synthesize_decision(G, q["query"], causal_path=path)
            conf = score_confidence(G, decision, causal_path=path, query_type="causal")
            # A causal "why" answer is held to a stricter recovery bar than a
            # factual lookup: it must recover most of the gold chain, not a
            # fragment. (Local to this harness; the factual judge is unchanged.)
            correct, recall = judge_correct(q, decision, tau=CAUSAL_TAU)
            rows.append({
                "id": q["id"], "k_steps": k,
                "path_completeness": conf.breakdown["path_completeness"]["value"],
                "confidence": conf.score, "band": conf.band,
                "correct": bool(correct), "gold_recall": round(recall, 3),
            })
    return rows


def causal_thesis(rows: list[dict]) -> dict:
    correct = [1.0 if r["correct"] else 0.0 for r in rows]
    pc = [r["path_completeness"] for r in rows]
    conf = [r["confidence"] for r in rows]
    return {
        "n_conditions": len(rows),
        "overall_accuracy": round(sum(correct) / len(rows), 3) if rows else 0.0,
        "pearson_path_completeness_vs_correct": round(_pearson(pc, correct), 3),
        "pearson_confidence_vs_correct": round(_pearson(conf, correct), 3),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(corpus: Corpus | None = None) -> dict:
    corpus = corpus or Corpus.load_named("incident")
    spec = load_causal_eval()
    queries = spec["queries"]

    # routing check
    routing = [{"id": q["id"], "expected": q["expected_type"],
                "routed": classify_query(q["query"]),
                "ok": classify_query(q["query"]) == q["expected_type"]}
               for q in queries]

    # 1. path retrieval (causal queries only)
    causal_qs = [q for q in queries if q.get("expected_type") == "causal"]
    pr = [path_retrieval_scores(corpus, q) for q in causal_qs]
    mean_p = round(sum(r["precision"] for r in pr) / len(pr), 3) if pr else 0.0
    mean_r = round(sum(r["recall"] for r in pr) / len(pr), 3) if pr else 0.0
    mean_f1 = round(sum(r["f1"] for r in pr) / len(pr), 3) if pr else 0.0

    # 2 + 3. conditions → calibration + thesis
    rows = generate_causal_conditions(corpus, causal_qs)
    bins, ece = calibration(rows)
    thesis = causal_thesis(rows)
    _plot(bins, ece)

    out = {
        "routing": routing,
        "path_retrieval": {"mean_precision": mean_p, "mean_recall": mean_r,
                           "mean_f1": mean_f1, "per_query": pr},
        "calibration": {"ece": round(ece, 4), "bins": bins},
        "thesis_causal": thesis,
        "conditions": rows,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    _print(out)
    return out


def _plot(bins, ece) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    xs = [b["mean_conf"] for b in bins]
    ys = [b["accuracy"] for b in bins]
    ax.plot(xs, ys, "-o", color="#9467bd")
    ax.set_xlabel("predicted confidence (bin mean)")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"Causal reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CALIB_PNG, dpi=120); plt.close(fig)


def _print(out: dict) -> None:
    print("=" * 68)
    print("CAUSAL EVAL — Stage 3.5 path retrieval + calibration + thesis")
    print("=" * 68)
    print("query routing (causal vs factual):")
    for r in out["routing"]:
        print(f"  [{'ok' if r['ok'] else 'XX'}] {r['id']:<4} {r['routed']:<8} "
              f"(expected {r['expected']})")
    pr = out["path_retrieval"]
    print(f"\npath-retrieval accuracy (gold path edges):")
    print(f"  mean precision={pr['mean_precision']}  recall={pr['mean_recall']}  "
          f"f1={pr['mean_f1']}")
    for r in pr["per_query"]:
        flag = "ok" if r["recall"] == 1.0 else "XX"
        print(f"  [{flag}] {r['id']:<4} P={r['precision']} R={r['recall']} "
              f"root={r['root_cause_recovered']} personas={r['personas_consistent']}")
    print(f"\ncalibration (causal): ECE = {out['calibration']['ece']}  "
          f"over {out['thesis_causal']['n_conditions']} conditions")
    t = out["thesis_causal"]
    print(f"\nthesis (causal): overall accuracy = {t['overall_accuracy']}")
    print(f"  Pearson r(path_completeness, correct) = "
          f"{t['pearson_path_completeness_vs_correct']:+.3f}")
    print(f"  Pearson r(confidence, correct)        = "
          f"{t['pearson_confidence_vs_correct']:+.3f}")
    print("=" * 68)
    print(f"wrote {OUT_PATH.name}, {CALIB_PNG.name}")


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
