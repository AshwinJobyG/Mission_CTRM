"""End-to-end tests exercising every module and key invariant.

Plain-assert tests (no pytest needed): `python -m tests.test_e2e` from the
project root. LLM/HF network paths are covered via fallbacks + a mocked ION call.
"""

from __future__ import annotations

import os

from src import decision as dec_mod
from src.access import (ROLE_CLEARANCES, can_see, filtered_corpus,
                        pipeline_for_role)
from src.confidence import WEIGHTS, score_confidence
from src.corpus import Corpus
from src.decision import (DecisionResult, decide_for_query, synthesize_decision)
from src.eval_confidence import (calibration, evaluate_bands,
                                 generate_conditions, judge_correct,
                                 load_queries, thesis_validation)
from src.graph_builder import (GraphBoostedRetriever, build_context_map,
                               measure_graph_lift)
from src.retrieval import build_retrievers

PASSED = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    PASSED.append(name)
    print(f"  ok  {name}")


def test_corpus():
    print("[corpus]")
    c = Corpus.load()
    check("33 nodes", len(c) == 33)
    check("2 dangling refs", len(c.dangling_refs()) == 2)
    check("2 contradiction pairs", len(c.contradiction_pairs()) == 2)
    check("2 supersedes pairs", len(c.supersedes_pairs()) == 2)
    check("3 restricted/hr nodes", len(c.by_security_label("restricted", "hr_only")) == 3)
    return c


def test_retrieval(c):
    print("[retrieval]")
    R = build_retrievers(c)
    check("three retrievers", set(R) == {"keyword", "embedding", "hybrid"})
    for name, r in R.items():
        out = r.retrieve("settlement batch failure root cause", k=5)
        check(f"{name}: returns <=5 (id,score)", len(out) <= 5 and all(
            isinstance(t, tuple) and t[0] in c for t in out))
    return R


def test_graph(c, R):
    print("[graph]")
    top = [n for n, _ in R["hybrid"].retrieve("root cause of SG settlement failures", 12)]
    G = build_context_map(c, top, query="q")
    check("DiGraph non-empty", G.number_of_nodes() > 0 and G.number_of_edges() > 0)
    for n in G:
        d = G.nodes[n]
        check_keys = {"hubness", "tier_w", "freshness", "status_w", "in_degree"}
        if not check_keys <= set(d):
            raise AssertionError(f"node {n} missing signals")
    PASSED.append("all nodes carry structural signals"); print("  ok  all nodes carry structural signals")
    check("typed edges", all("rel" in G.edges[e] for e in G.edges))
    check("contradictions exposed", isinstance(G.graph["contradictions"], list))
    check("dangling exposed", isinstance(G.graph["dangling"], list))
    lift = measure_graph_lift(c, k=5)
    dp = lift["hybrid+graph"]["p_at_k"] - lift["hybrid"]["p_at_k"]
    check("graph lift computed (non-negative P@5)", dp >= 0)
    print(f"      graph P@5 lift = {dp:+.3f}")
    return G


def test_decision(c, G):
    print("[decision]")
    res = synthesize_decision(G, "root cause?")
    check("returns DecisionResult", isinstance(res, DecisionResult))
    check("all citations in subgraph", all(cid in G for cid in res.cited_node_ids))
    check("no hallucinated citations (extractive)", res.hallucinated_citations == [])
    check("method is a known backend", res.method in {"extractive", "ion-llm", "anthropic"})
    return res


def test_ion_mock(c, G):
    """Exercise the ION synthesis + JSON-parse + citation-validation path."""
    print("[decision: mocked ION]")

    class _Resp:
        content = ('{"decision_text": "Root cause is connection-pool exhaustion '
                   '[RES-12], also cited [FAKE-999].", '
                   '"cited_node_ids": ["RES-12", "FAKE-999"], '
                   '"noted_gaps": ["upstream cause unknown"]}')

    class _LLM:
        def invoke(self, messages):
            assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
            return _Resp()

    saved = (os.environ.get("ION_LLM_API_URL"), os.environ.get("ION_LLM_API_KEY"),
             os.environ.get("ION_LLM_MODEL"), dec_mod._ion_client)
    os.environ.update(ION_LLM_API_URL="http://mock", ION_LLM_API_KEY="k",
                      ION_LLM_MODEL="m")
    dec_mod._ion_client = lambda: _LLM()
    try:
        assert "RES-12" in G  # ensure the valid citation is present
        res = synthesize_decision(G, "root cause?")
        check("ION backend selected", res.method == "ion-llm")
        check("valid citation kept", "RES-12" in res.cited_node_ids)
        check("fake citation flagged hallucinated", "FAKE-999" in res.hallucinated_citations)
        check("model gaps parsed", res.model_noted_gaps == ["upstream cause unknown"])
    finally:
        dec_mod._ion_client = saved[3]
        for k, v in zip(("ION_LLM_API_URL", "ION_LLM_API_KEY", "ION_LLM_MODEL"), saved[:3]):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_confidence(c, G, res):
    print("[confidence]")
    conf = score_confidence(G, res)
    check("breakdown has 6 features", set(conf.breakdown) == set(WEIGHTS))
    check("weights match", all(conf.breakdown[f]["weight"] == WEIGHTS[f] for f in WEIGHTS))
    check("score reconstructs (subtotal x sufficiency)",
          abs(round(conf.reconstruct(), 4) - conf.score) <= 1e-4)
    check("score in [0,1]", 0.0 <= conf.score <= 1.0)
    check("band valid", conf.band in {"high", "medium", "low"})
    check("gap_report is structured", all("type" in g and "detail" in g for g in conf.gap_report))


def test_access(c):
    print("[access]")
    intern = filtered_corpus(c, "intern")
    lead = filtered_corpus(c, "lead")
    check("intern sees fewer than lead", len(intern) < len(lead))
    restricted = [n["id"] for n in c if n["security_label"] in ("restricted", "hr_only")]
    check("restricted nodes absent from intern view", all(r not in intern for r in restricted))
    check("lead can see RES-13 (restricted)", "RES-13" in lead)
    check("intern cannot see RES-13", "RES-13" not in intern)
    # forbidden nodes never enter the subgraph
    G_i, _, conf_i = pipeline_for_role(c, "root cause of SG settlement failures?", "intern")
    G_l, _, conf_l = pipeline_for_role(c, "root cause of SG settlement failures?", "lead")
    check("no restricted node in intern subgraph", all(r not in G_i for r in restricted))
    check("intern confidence <= lead confidence", conf_i.score <= conf_l.score)
    print(f"      intern={conf_i.score:.3f}({conf_i.band})  lead={conf_l.score:.3f}({conf_l.band})")


def test_eval_harness(c):
    print("[eval harness]")
    queries = load_queries()
    bands = evaluate_bands(c, queries)
    check("band eval one row per query", len(bands) == len(queries))
    rows = generate_conditions(c, queries, roles=("intern", "lead"), n_seeds_list=(6, 10))
    check(">=30 conditions generated", len(rows) >= 30)
    _, ece = calibration(rows)
    check("ECE in [0,1]", 0.0 <= ece <= 1.0)
    tv = thesis_validation(rows)
    r = tv["pearson_r_with_correctness"]["confidence"]
    check("confidence<->correctness correlation computed", -1.0 <= r <= 1.0)
    print(f"      conditions={len(rows)}  ECE={ece:.3f}  conf~correct r={r:+.3f}")
    # correctness judge sanity
    ok, recall = judge_correct(queries[0], decide_for_query(c, queries[0]["query"])[1])
    check("judge returns (bool, recall)", isinstance(ok, bool) and 0.0 <= recall <= 1.0)


def main():
    c = test_corpus()
    R = test_retrieval(c)
    G = test_graph(c, R)
    res = test_decision(c, G)
    test_ion_mock(c, G)
    test_confidence(c, G, res)
    test_access(c)
    test_eval_harness(c)
    print(f"\nALL {len(PASSED)} CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
