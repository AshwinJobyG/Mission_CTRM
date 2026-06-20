"""Stage 3.5 — causal traversal (inserted between CONTEXT MAP and THINK).

The hard design rule of this integration: **the graph traversal does the
structural reasoning; the LLM only narrates the path the traversal already
found.** For a causal ("why") question this module walks the knowledge graph
backward from the impact to its root cause(s) — *before* any LLM call — and
emits an explicit, inspectable :class:`CausalPath`. THINK is then constrained to
narrate that chain and cite the evidence on each step (see ``decision.py``).

Pipeline position::

    SEARCH → CONTEXT MAP → [Stage 3.5: causal traversal] → THINK → CONFIDENCE

Factual questions never trigger traversal — they keep the existing THINK path
exactly (``CausalPath`` is empty / ``None``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Corpus
from .schema import CAUSAL_TRAVERSAL_RELS, node_type

# ---------------------------------------------------------------------------
# 1. Question routing  (causal vs factual)
# ---------------------------------------------------------------------------

_CAUSAL_RE = re.compile(
    r"\b(why|root[\s-]?cause|caused?|causing|because|led to|lead to|leads to|"
    r"reason|result(?:ed)? in|what went wrong|how did .*(?:happen|occur)|"
    r"trace|stem(?:med)? from|due to|driver|contribut\w*)\b",
    re.IGNORECASE,
)

_NODE_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]+-[A-Za-z0-9]+\b")


def classify_query(query: str | None) -> str:
    """Tag a query as ``"causal"`` or ``"factual"`` (keyword/intent routing).

    Deterministic and offline by design. Only ``causal`` queries trigger
    traversal; everything else keeps the existing factual THINK path.
    """
    return "causal" if (query and _CAUSAL_RE.search(query)) else "factual"


def is_causal(query: str | None) -> bool:
    return classify_query(query) == "causal"


# ---------------------------------------------------------------------------
# 2. Entry-point resolution
# ---------------------------------------------------------------------------

def resolve_entry_node(corpus: Corpus, query: str | None) -> str | None:
    """Map the question to the node the traversal starts from.

    Priority: an explicit node id named in the query → the incident node →
    the highest-id ticket of type incident → ``None`` (no entry point).
    """
    if query:
        for tok in _NODE_ID_RE.findall(query):
            if tok in corpus:
                return tok
    incidents = [n["id"] for n in corpus if node_type(n) == "incident"]
    if incidents:
        return sorted(incidents)[0]
    return None


# ---------------------------------------------------------------------------
# 3+4. Backward traversal → CausalPath
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalStep:
    """One (node → edge → node) hop on the causal path, with its evidence."""

    src: str
    rel: str
    target: str
    src_title: str
    target_title: str
    target_type: str
    depth: int
    evidence: tuple | None  # (source_id, passage) or None  → a gap if None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> dict:
        return {
            "src": self.src, "rel": self.rel, "target": self.target,
            "src_title": self.src_title, "target_title": self.target_title,
            "target_type": self.target_type, "depth": self.depth,
            "evidence": ({"source_id": self.evidence[0], "passage": self.evidence[1]}
                         if self.evidence else None),
        }


# Persona roles surfaced by the traversal (who raised the risk / made the call /
# owned the code), keyed by the relation that reveals them.
_PERSONA_BY_REL = {
    "PRIORITIZED_BY": "made_the_call",
    "OWNED_BY": "owned_the_code",
}


@dataclass
class CausalPath:
    """An ordered, inspectable root-cause→impact path (first-class output).

    Serializable and human-readable. Fed to THINK as the spine, rendered in the
    UI, and scored by the path-completeness confidence feature.
    """

    entry: str | None
    steps: list[CausalStep] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    personas: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.steps)

    def nodes(self) -> list[str]:
        seen: list[str] = []
        for s in ([self.entry] if self.entry else []):
            if s and s not in seen:
                seen.append(s)
        for st in self.steps:
            for nid in (st.src, st.target):
                if nid not in seen:
                    seen.append(nid)
        return seen

    def edges(self) -> list[tuple[str, str, str]]:
        return [(s.src, s.rel, s.target) for s in self.steps]

    def evidence_sources(self) -> list[str]:
        """Distinct node ids cited as evidence across the path."""
        out: list[str] = []
        for s in self.steps:
            if s.evidence and s.evidence[0] and s.evidence[0] not in out:
                out.append(s.evidence[0])
        return out

    def context_ids(self) -> list[str]:
        """All ids the narration may cite: path nodes ∪ evidence sources.

        The context map must contain these so the decision's citations validate
        (rather than being flagged as hallucinated against a too-narrow subgraph).
        """
        ids = self.nodes()
        for sid in self.evidence_sources():
            if sid not in ids:
                ids.append(sid)
        return ids

    @property
    def evidenced_steps(self) -> int:
        return sum(1 for s in self.steps if s.has_evidence)

    @property
    def unevidenced(self) -> list[CausalStep]:
        return [s for s in self.steps if not s.has_evidence]

    def to_dict(self) -> dict:
        return {
            "entry": self.entry,
            "root_causes": self.root_causes,
            "personas": self.personas,
            "n_steps": len(self.steps),
            "evidenced_steps": self.evidenced_steps,
            "steps": [s.to_dict() for s in self.steps],
        }

    def render(self) -> str:
        """Human-readable step-by-step (used in the prompt spine and the CLI)."""
        if not self.steps:
            return "(no causal path traced)"
        lines = [f"CAUSAL PATH traced from impact [{self.entry}] to root cause(s) "
                 f"{self.root_causes}:"]
        for i, s in enumerate(self.steps, 1):
            ev = (f'  evidence [{s.evidence[0]}]: "{s.evidence[1]}"'
                  if s.evidence else "  evidence: (MISSING — gap)")
            lines.append(f"  {i}. [{s.src}] --{s.rel}--> [{s.target}] "
                         f"({s.target_type}: {s.target_title})\n{ev}")
        if self.personas:
            who = ", ".join(f"{role}={nid}" for role, nid in self.personas.items())
            lines.append(f"  personas: {who}")
        return "\n".join(lines)


def trace_causal_path(
    corpus: Corpus, query: str | None = None, *, entry: str | None = None
) -> CausalPath:
    """Walk the causal subgraph from the impact node out to root cause(s).

    Follows only relations in :data:`CAUSAL_TRAVERSAL_RELS`, in author-direction
    (impact → cause), breadth-first, so the emitted order reads impact-first and
    terminates at root-cause leaves. Each hop carries the edge's evidence; a hop
    with no evidence is preserved as a step (a scored gap), never dropped.
    """
    entry = entry or resolve_entry_node(corpus, query)
    if entry is None or entry not in corpus:
        return CausalPath(entry=entry)

    # Adjacency over causal-traversal relations only (skip dangling targets).
    adj: dict[str, list] = {}
    for e in corpus.edges():
        if e.rel in CAUSAL_TRAVERSAL_RELS and not corpus.is_dangling(e):
            adj.setdefault(e.src, []).append(e)

    steps: list[CausalStep] = []
    enqueued = {entry}
    queue: list[tuple[str, int]] = [(entry, 0)]
    reached: set[str] = {entry}
    while queue:
        nid, depth = queue.pop(0)
        for e in sorted(adj.get(nid, []), key=lambda x: (x.rel, x.target)):
            tgt = corpus.get(e.target) or {}
            steps.append(CausalStep(
                src=e.src, rel=e.rel, target=e.target,
                src_title=(corpus.get(e.src) or {}).get("title", e.src),
                target_title=tgt.get("title", e.target),
                target_type=node_type(tgt) if tgt else "unknown",
                depth=depth, evidence=e.evidence,
            ))
            reached.add(e.target)
            if e.target not in enqueued:
                enqueued.add(e.target)
                queue.append((e.target, depth + 1))

    # Root causes: the Risk node(s) and any causal leaf (no outgoing causal edge)
    # that is a ticket/risk — i.e. the deepest "why" the graph offers.
    root_causes: list[str] = []
    for nid in reached:
        n = corpus.get(nid) or {}
        nt = node_type(n)
        has_out = bool(adj.get(nid))
        if nt == "risk" or (nt == "ticket" and not has_out):
            root_causes.append(nid)

    # Personas: who raised the risk / made the call / owned the code.
    personas: dict[str, str] = {}
    for s in steps:
        if s.rel in _PERSONA_BY_REL and s.target_type == "person":
            personas.setdefault(_PERSONA_BY_REL[s.rel], s.target)
        elif s.rel == "RAISED_RISK" and s.target_type == "person":
            personas.setdefault("raised_risk", s.target)

    return CausalPath(entry=entry, steps=steps,
                      root_causes=sorted(root_causes), personas=personas)


# ---------------------------------------------------------------------------
# Unified pipeline: SEARCH → CONTEXT MAP → Stage 3.5 → THINK → CONFIDENCE
# ---------------------------------------------------------------------------

@dataclass
class CausalAnswer:
    """Everything the UI / eval need for one (possibly causal) query."""

    query: str
    query_type: str            # "causal" | "factual"
    graph: object              # nx.DiGraph context map
    decision: object           # DecisionResult
    confidence: object         # ConfidenceResult
    causal_path: CausalPath | None
    seeds: list[str]


def answer_query(
    corpus: Corpus, query: str, *, role: str | None = None,
    retriever=None, n_seeds: int = 8,
) -> CausalAnswer:
    """Run the full read path, inserting Stage 3.5 for causal questions.

    Factual queries are handled exactly as before (no traversal, empty path).
    Causal queries trace the path first, fold its nodes + evidence sources into
    the context map so citations validate, then constrain THINK to narrate it.
    Lazy imports keep this module free of import cycles.
    """
    from .access import filtered_corpus
    from .confidence import score_confidence
    from .decision import synthesize_decision
    from .graph_builder import GraphBoostedRetriever, build_context_map
    from .retrieval import build_retrievers

    view = filtered_corpus(corpus, role) if role else corpus
    if retriever is None:
        retriever = GraphBoostedRetriever(build_retrievers(view)["hybrid"], view)

    qtype = classify_query(query)
    retrieved = [nid for nid, _ in retriever.retrieve(query, k=n_seeds)]

    causal_path: CausalPath | None = None
    seeds = list(retrieved)
    if qtype == "causal":
        # Trace over the (role-filtered) full graph: the structural reasoning is
        # done by the graph, independent of what retrieval happened to surface.
        causal_path = trace_causal_path(view, query)
        for nid in causal_path.context_ids():
            if nid in view and nid not in seeds:
                seeds.append(nid)

    G = build_context_map(view, seeds, query=query)
    decision = synthesize_decision(G, query, causal_path=causal_path)
    conf = score_confidence(G, decision, causal_path=causal_path, query_type=qtype)
    return CausalAnswer(query=query, query_type=qtype, graph=G, decision=decision,
                        confidence=conf, causal_path=causal_path, seeds=seeds)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main() -> int:
    corpus = Corpus.load_named("incident")
    for q in ("Why did NGPOWER-145 become a P0 incident?",
              "Who owns the EPEX connectivity product?"):
        qtype = classify_query(q)
        print("=" * 72)
        print(f"query: {q}\nrouted as: {qtype.upper()}")
        print("-" * 72)
        if qtype == "causal":
            path = trace_causal_path(corpus, q)
            print(path.render())
        else:
            print("(factual — no traversal; existing THINK path is used unchanged)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
