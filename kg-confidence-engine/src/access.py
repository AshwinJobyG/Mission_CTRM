"""Role-based access control enforced as sub-graph filtering.

Security is enforced in CODE at the earliest layer — the corpus/retrieval
boundary — not by asking the LLM nicely. A node the user's role may not see is
removed from the corpus view entirely, so it never enters the candidate set, the
context map, the LLM prompt, or the confidence computation. This mirrors the
reference model's "the query engine literally cannot see the node", and unlike
prompt-level access control it is not prompt-injectable.

A useful emergent property: when a restricted node is removed for a role, any
link that pointed at it becomes a *dangling reference* for that role — so the
role automatically reasons over an acknowledged-incomplete picture, which the
confidence model already penalizes and the gap report already surfaces.
"""

from __future__ import annotations

from .confidence import ConfidenceResult, score_confidence
from .corpus import Corpus
from .decision import DecisionResult, synthesize_decision
from .graph_builder import GraphBoostedRetriever, build_context_map
from .retrieval import build_retrievers

# Mocked roles -> the set of security labels each role is cleared to see.
ROLE_CLEARANCES: dict[str, set[str]] = {
    "intern": {"public"},
    "engineer": {"public", "internal"},
    "lead": {"public", "internal", "restricted"},
    "hr": {"public", "internal", "hr_only"},
}
DEFAULT_ROLE = "lead"


def can_see(node: dict, role: str) -> bool:
    clearance = ROLE_CLEARANCES.get(role)
    if clearance is None:
        raise ValueError(f"unknown role: {role!r} (known: {sorted(ROLE_CLEARANCES)})")
    return node["security_label"] in clearance


def filter_by_role(nodes, role: str) -> list[dict]:
    """Drop nodes the role is not cleared to see. Applied at the retrieval boundary."""
    return [n for n in nodes if can_see(n, role)]


def filtered_corpus(corpus: Corpus, role: str) -> Corpus:
    """A corpus view containing only the nodes visible to ``role``."""
    visible = filter_by_role(list(corpus), role)
    return Corpus(visible, corpus.meta)


def pipeline_for_role(
    corpus: Corpus, query: str, role: str, *, n_seeds: int = 8
) -> tuple[object, DecisionResult, ConfidenceResult]:
    """Full read path under a role: filter -> retrieve -> graph -> decide -> score."""
    view = filtered_corpus(corpus, role)
    retriever = GraphBoostedRetriever(build_retrievers(view)["hybrid"], view)
    top = [nid for nid, _ in retriever.retrieve(query, k=n_seeds)]
    G = build_context_map(view, top, query=query)
    decision = synthesize_decision(G, query)
    conf = score_confidence(G, decision)
    return G, decision, conf


def _demo(corpus: Corpus, query: str, roles=("intern", "engineer", "lead")) -> None:
    print("=" * 72)
    print(f"ACCESS CONTRAST — query: {query}")
    print("=" * 72)
    print(f"corpus visibility by role: "
          + ", ".join(f"{r}={len(filtered_corpus(corpus, r))}" for r in ROLE_CLEARANCES))
    print()
    for role in roles:
        G, decision, conf = pipeline_for_role(corpus, query, role)
        hidden = [n["id"] for n in corpus if not can_see(n, role)]
        print(f"--- role: {role} ---")
        print(f"  visible nodes: {len(filtered_corpus(corpus, role))}  | "
              f"hidden: {len(hidden)} ({', '.join(hidden) or 'none'})")
        print(f"  context map: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  cited: {decision.cited_node_ids}")
        print(f"  confidence: {conf.score:.3f} -> {conf.band.upper()}")
        gaps = [g['detail'] for g in conf.gap_report if g['type'] == 'dangling_reference']
        print(f"  dangling gaps: {gaps or 'none'}")
        print()


if __name__ == "__main__":
    corpus = Corpus.load()
    # A query whose key evidence is restricted: the customer/SLA-credit impact
    # lives in restricted JIRA-4416 — visible to a lead, invisible to an intern.
    _demo(corpus, "what was the financial and SLA-credit impact of the SG settlement outage on the customer?")
    # And the root-cause query, where restricted RES-13 (vendor RCA) is extra
    # corroboration a lead sees but an intern does not.
    _demo(corpus, "what is the root cause of the SG settlement batch failures?")
