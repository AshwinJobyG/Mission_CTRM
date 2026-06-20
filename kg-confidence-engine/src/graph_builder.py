"""Context-map builder + per-node structural signals.

Turns a retrieved reference set into the interrelated context map (a
``networkx.DiGraph`` with typed edges) and computes, *relative to that subgraph
and query*, the structural signals that feed both retrieval re-ranking (the
graph "lift") and the confidence model (Phase 5):

* hub-ness / adjacency — in-degree within the subgraph (how many other retrieved
  references point at this node). A hub is a corroboration anchor for the query.
* source-tier weight — resolution > runbook/doc > ticket/incident > comment.
* freshness — exponential decay of node age relative to the newest node present.
* status weight — verified/resolved > in_progress > open > wontfix > deprecated.

It also exposes the subgraph's contradiction pairs and dangling references as
structured gap signals.

Run: ``python -m src.graph_builder``  (prints a sample subgraph with signals and
the measured P@5 lift of hybrid+graph over hybrid).
"""

from __future__ import annotations

import math
from datetime import date

import networkx as nx

from .corpus import Corpus
from .retrieval import build_retrievers
from .schema import DEFAULT_NODE_TYPE

# Numeric weights for the structural signals.
TIER_WEIGHT = {
    "resolution": 1.0,
    "runbook": 0.8,
    "doc": 0.7,
    "incident": 0.6,
    "ticket": 0.5,
    "comment": 0.3,
}
STATUS_WEIGHT = {
    "verified": 1.0,
    "resolved": 0.9,
    "in_progress": 0.5,
    "open": 0.4,
    "wontfix": 0.2,
    "deprecated": 0.1,
}
FRESHNESS_TAU_DAYS = 180.0  # ~6 months half-life-ish decay


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def build_context_map(
    corpus: Corpus, retrieved_ids: list[str], *, query: str | None = None
) -> nx.DiGraph:
    """Build the retrieved subgraph with typed edges and per-node signals.

    Node set = the retrieved (seed) ids plus any one-hop link targets that exist
    in the corpus. Edges = every declared link whose source and target are both
    in the node set, carrying its relation as an edge attribute. Dangling links
    (target absent from the corpus) are recorded as a graph-level gap signal.
    """
    seeds = [nid for nid in retrieved_ids if nid in corpus]
    nodes: set[str] = set(seeds)
    for nid in seeds:
        for link in corpus.nodes[nid].get("links", []):
            if link["target"] in corpus:
                nodes.add(link["target"])

    G = nx.DiGraph()
    G.graph["query"] = query
    G.graph["seeds"] = list(seeds)

    for nid in nodes:
        n = corpus.nodes[nid]
        # ``.get`` defaults keep record nodes byte-identical (they always carry
        # these fields) while letting typed entity nodes — which may omit
        # body/status/etc. — flow through without a KeyError (Phase B/C).
        G.add_node(
            nid,
            type=n.get("type", "record"),
            node_type=n.get("node_type", DEFAULT_NODE_TYPE),
            title=n["title"],
            body=n.get("body", ""),
            status=n.get("status", "open"),
            source_tier=n.get("source_tier", "ticket"),
            date=n["date"] if "date" in n else None,
            author=n.get("author", ""),
            security_label=n.get("security_label", "internal"),
            seed=(nid in set(seeds)),
        )

    dangling: list[tuple[str, str, str]] = []
    for nid in nodes:
        for link in corpus.nodes[nid].get("links", []):
            tgt, rel = link["target"], link["rel"]
            if tgt in nodes:
                G.add_edge(nid, tgt, rel=rel)
            elif tgt not in corpus:
                dangling.append((nid, rel, tgt))

    _attach_signals(G)
    G.graph["dangling"] = dangling
    G.graph["contradictions"] = subgraph_contradictions(G)
    return G


def _attach_signals(G: nx.DiGraph) -> None:
    if not G:
        return
    in_degrees = dict(G.in_degree())
    max_in = max(in_degrees.values()) if in_degrees else 0
    newest = max(_parse(G.nodes[n]["date"]) for n in G)

    for n in G:
        nd = G.nodes[n]
        indeg = in_degrees.get(n, 0)
        nd["in_degree"] = indeg
        nd["hubness"] = (indeg / max_in) if max_in > 0 else 0.0
        nd["tier_w"] = TIER_WEIGHT.get(nd["source_tier"], 0.4)
        nd["status_w"] = STATUS_WEIGHT.get(nd["status"], 0.4)
        age_days = (newest - _parse(nd["date"])).days
        nd["freshness"] = math.exp(-age_days / FRESHNESS_TAU_DAYS)


def subgraph_contradictions(G: nx.DiGraph) -> list[tuple[str, str]]:
    """Unordered node pairs joined by a ``contradicts`` edge within the subgraph."""
    seen: set[frozenset[str]] = set()
    pairs: list[tuple[str, str]] = []
    for u, v, data in G.edges(data=True):
        if data.get("rel") == "contradicts":
            key = frozenset((u, v))
            if key not in seen:
                seen.add(key)
                pairs.append((u, v))
    return pairs


def subgraph_dangling(G: nx.DiGraph) -> list[tuple[str, str, str]]:
    """Dangling references (target absent from corpus) recorded for this subgraph."""
    return list(G.graph.get("dangling", []))


# ============================================================================
# Graph-adjacency re-rank (the "graph lift")
# ============================================================================

class GraphBoostedRetriever:
    """Re-rank a hybrid candidate pool by each node's hub-ness in the subgraph.

    A node that many other retrieved references point at is a corroboration
    anchor for this query, so its score is boosted. This is the empirical
    version of GBrain's graph-over-vector lift, measured against plain hybrid.
    """

    def __init__(self, hybrid, corpus: Corpus, *, pool: int = 12, alpha: float = 0.6):
        self.hybrid = hybrid
        self.corpus = corpus
        self.pool = pool
        self.alpha = alpha
        self.name = "hybrid + graph-boost"

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        base = self.hybrid.scores(query)
        candidates = [nid for nid, _ in sorted(base.items(), key=lambda kv: -kv[1])[: self.pool]]
        G = build_context_map(self.corpus, candidates, query=query)

        vals = [base[c] for c in candidates]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        boosted: dict[str, float] = {}
        for c in candidates:
            base_norm = (base[c] - lo) / span
            hub = G.nodes[c]["hubness"] if c in G else 0.0
            boosted[c] = base_norm + self.alpha * hub
        ranked = sorted(boosted.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


def measure_graph_lift(corpus: Corpus, k: int = 5) -> dict:
    """Measure P@k/R@k of hybrid vs hybrid+graph-boost on the eval set."""
    from .eval_retrieval import evaluate, load_queries

    retrievers = build_retrievers(corpus)
    boosted = GraphBoostedRetriever(retrievers["hybrid"], corpus)
    subset = {"hybrid": retrievers["hybrid"], "hybrid+graph": boosted}
    return evaluate(corpus, subset, load_queries(), k=k)


# ============================================================================
# CLI demo
# ============================================================================

def _print_subgraph(corpus: Corpus, query: str) -> None:
    retrievers = build_retrievers(corpus)
    top = [nid for nid, _ in retrievers["hybrid"].retrieve(query, k=12)]
    G = build_context_map(corpus, top, query=query)
    print(f"query: {query}")
    print(f"subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    print(f"{'node':10} {'tier':>5} {'fresh':>6} {'stat':>5} {'in':>3} {'hub':>5}  title")
    print("-" * 92)
    for n in sorted(G, key=lambda x: -G.nodes[x]["hubness"]):
        d = G.nodes[n]
        mark = "*" if d["seed"] else " "
        print(
            f"{mark}{n:9} {d['tier_w']:>5.2f} {d['freshness']:>6.2f} "
            f"{d['status_w']:>5.2f} {d['in_degree']:>3} {d['hubness']:>5.2f}  {d['title'][:40]}"
        )
    print(f"\ncontradiction pairs in subgraph: {G.graph['contradictions']}")
    print(f"dangling references in subgraph: {G.graph['dangling']}")


def main() -> int:
    corpus = Corpus.load()
    _print_subgraph(corpus, "what is the root cause of the SG settlement batch failures?")

    print("\n" + "=" * 64)
    print("GRAPH LIFT — hybrid vs hybrid+graph-boost")
    print("=" * 64)
    results = measure_graph_lift(corpus, k=5)
    print(f"{'retriever':<24} {'P@5':>8} {'R@5':>8}")
    print("-" * 64)
    for key in ("hybrid", "hybrid+graph"):
        r = results[key]
        print(f"{r['name']:<24} {r['p_at_k']:>8.4f} {r['r_at_k']:>8.4f}")
    dp = results["hybrid+graph"]["p_at_k"] - results["hybrid"]["p_at_k"]
    dr = results["hybrid+graph"]["r_at_k"] - results["hybrid"]["r_at_k"]
    print("-" * 64)
    print(f"{'lift':<24} {dp:>+8.4f} {dr:>+8.4f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
