"""Decision synthesis — the `think` stage, kept distinct from `search`.

Reasons over the context map to produce a grounded decision with inline node-ID
citations. Deliberate design choices (stated in the README):

* The model is grounded *only* in the provided subgraph nodes and must cite node
  IDs for every claim and flag what it cannot determine.
* The model does NOT output a confidence number — confidence is computed by us
  from graph structure (Phase 5), never self-reported by the LLM.

Backend: a single Anthropic API call (``claude-sonnet-4-6``), with the key read
from ``ANTHROPIC_API_KEY`` (never hardcoded). When no key is present we fall back
to a deterministic extractive synthesizer so the full pipeline (confidence,
access, UI) stays runnable; the fallback is clearly labeled via
``DecisionResult.method``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import networkx as nx

from .corpus import Corpus
from .graph_builder import build_context_map
from .retrieval import build_retrievers

MODEL = "claude-sonnet-4-6"
NODE_ID_RE = re.compile(r"\b[A-Z]{2,5}-\d+\b")
BODY_SNIPPET = 240


@dataclass
class DecisionResult:
    decision_text: str
    cited_node_ids: list[str]
    model_noted_gaps: list[str]
    method: str = "anthropic"
    hallucinated_citations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context serialization
# ---------------------------------------------------------------------------

def serialize_context_map(G: nx.DiGraph) -> str:
    """Compact, structured context block: nodes + explicit typed edges."""
    lines = ["NODES:"]
    for n in sorted(G, key=lambda x: -G.nodes[x].get("hubness", 0.0)):
        d = G.nodes[n]
        snippet = " ".join(d["body"].split())[:BODY_SNIPPET]
        lines.append(
            f"[{n} | {d['type']} | {d['status']} | tier={d['source_tier']} | {d['date']}] "
            f"{d['title']} — {snippet}"
        )
    lines.append("\nRELATIONSHIPS (typed edges):")
    for u, v, data in G.edges(data=True):
        lines.append(f"{u} --{data.get('rel')}--> {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful incident-analysis assistant. You are given a set of "
    "knowledge-graph NODES (each with an ID) and the typed RELATIONSHIPS between "
    "them. Answer the user's question using ONLY the information in these nodes.\n\n"
    "Rules:\n"
    "1. Ground every statement in the provided nodes. Do not use outside knowledge.\n"
    "2. Cite the node ID(s) inline in square brackets for every claim, e.g. [RES-12].\n"
    "3. Only cite IDs that appear in the provided NODES.\n"
    "4. Explicitly state anything you cannot determine from the provided context "
    "(missing links, contradictions, stale/superseded info).\n"
    "5. Do NOT output any confidence score, percentage, or certainty level — "
    "confidence is computed separately from the graph structure.\n\n"
    "Respond as strict JSON with keys: \"decision_text\" (string, with inline "
    "[ID] citations), \"cited_node_ids\" (list of the node IDs you cited), and "
    "\"noted_gaps\" (list of short strings describing what could not be "
    "determined). Output only the JSON object."
)

# Appended only for causal ("why") questions, when Stage 3.5 has already traced
# the path. The model NARRATES the traced chain — it does not re-derive it.
CAUSAL_SYSTEM_ADDENDUM = (
    "\n\nThis is a CAUSAL question. A CAUSAL PATH has already been traced from "
    "the knowledge graph and is provided below the context map. Your job is to "
    "NARRATE that specific chain from root cause to impact in prose.\n"
    "Additional rules:\n"
    "6. Follow the traced path exactly. Do NOT introduce any causal link that is "
    "not in the provided path.\n"
    "7. For every step, cite the step's node IDs and quote/reference the evidence "
    "passage attached to that step.\n"
    "8. Name the people on the path by their role (who raised the risk, who made "
    "the call, who owned the code) where the path identifies them."
)


def synthesize_decision(G: nx.DiGraph, query: str, *, causal_path=None) -> DecisionResult:
    """Produce a grounded, cited decision over the context map.

    Backend selection (env ``KGCE_LLM_BACKEND``, default ``auto``):
      auto -> ion (company-hosted, OpenAI-compatible) if configured,
              else anthropic if ANTHROPIC_API_KEY set, else extractive.
    Any backend failure falls through to the deterministic extractive fallback.

    ``causal_path`` (Stage 3.5): when present, THINK narrates that traced chain
    and cites the evidence on each step. When absent, behaviour is unchanged.
    """
    backend = os.environ.get("KGCE_LLM_BACKEND", "auto").lower()
    order = ["ion", "anthropic", "extractive"] if backend == "auto" else [backend]
    for b in order:
        try:
            if b == "ion" and _ion_configured():
                return _synthesize_ion(G, query, causal_path)
            if b == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                return _synthesize_anthropic(G, query, causal_path)
            if b == "extractive":
                return _synthesize_extractive(G, query, causal_path)
        except Exception as exc:  # network/SDK error -> fall back, but be loud
            print(f"[decision] {b} backend failed ({type(exc).__name__}: {exc}); "
                  "falling through.")
    return _synthesize_extractive(G, query, causal_path)


def build_messages(G: nx.DiGraph, query: str, causal_path=None) -> tuple[str, str]:
    """Shared (system, user) prompt for every LLM backend.

    For causal queries the traced path is prepended as the spine and the system
    prompt gains the narration constraints; factual queries are unchanged.
    """
    context = serialize_context_map(G)
    if causal_path:
        system = SYSTEM_PROMPT + CAUSAL_SYSTEM_ADDENDUM
        user = (f"QUESTION: {query}\n\nCONTEXT MAP:\n{context}\n\n"
                f"{causal_path.render()}")
        return system, user
    return SYSTEM_PROMPT, f"QUESTION: {query}\n\nCONTEXT MAP:\n{context}"


def _result_from_text(G, text: str, method: str) -> DecisionResult:
    data = _parse_json(text)
    decision_text = data.get("decision_text", text)
    cited = data.get("cited_node_ids") or NODE_ID_RE.findall(decision_text)
    gaps = data.get("noted_gaps", [])
    return _finalize(G, decision_text, cited, gaps, method=method)


def _synthesize_anthropic(G: nx.DiGraph, query: str, causal_path=None) -> DecisionResult:
    from anthropic import Anthropic

    client = Anthropic()
    system, user = build_messages(G, query, causal_path)
    msg = client.messages.create(
        model=MODEL, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    return _result_from_text(G, text, "anthropic")


# ---- company-hosted, OpenAI-compatible LLM (ION) ----------------------------

def _ion_configured() -> bool:
    return all(os.environ.get(k) for k in
               ("ION_LLM_API_URL", "ION_LLM_API_KEY", "ION_LLM_MODEL"))


def _ion_client():
    """ChatOpenAI pointed at the company-hosted endpoint (mirrors the supplied
    integration script: /v1 base, SSL verification configurable)."""
    import httpx
    from langchain_openai import ChatOpenAI

    base = os.environ["ION_LLM_API_URL"].rstrip("/")
    verify = os.environ.get("KGCE_ION_VERIFY_SSL", "false").lower() in ("1", "true", "yes")
    return ChatOpenAI(
        base_url=f"{base}/v1",
        api_key=os.environ["ION_LLM_API_KEY"],
        model=os.environ["ION_LLM_MODEL"],
        max_tokens=int(os.environ.get("KGCE_ION_MAX_TOKENS", "4000")),
        temperature=0,
        http_client=httpx.Client(verify=verify),
    )


def _synthesize_ion(G: nx.DiGraph, query: str, causal_path=None) -> DecisionResult:
    llm = _ion_client()
    system, user = build_messages(G, query, causal_path)
    resp = llm.invoke([{"role": "system", "content": system},
                       {"role": "user", "content": user}])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _result_from_text(G, text, "ion-llm")


def _synthesize_extractive(G: nx.DiGraph, query: str, causal_path=None) -> DecisionResult:
    """Deterministic, grounded fallback: rank support by structural signal.

    For a causal query with a traced path, narrate that path step-by-step with
    per-step evidence citations (so the offline/default demo also shows the
    "traversal finds it, narration cites it" behaviour).
    """
    if causal_path:
        return _synthesize_extractive_causal(G, query, causal_path)
    if not G:
        return DecisionResult("No context could be retrieved for this query.", [], [],
                              method="extractive")

    # Cite the query-relevant seeds (the nodes actually retrieved for this
    # query), in retrieval order — NOT globally-high-signal nodes pulled in by
    # graph expansion. This keeps the decision (and thus the confidence computed
    # over it) faithful to each query's real evidential basis.
    seeds = [s for s in G.graph.get("seeds", []) if s in G]
    if not seeds:  # degenerate: fall back to structural ranking
        def support_score(n: str) -> float:
            d = G.nodes[n]
            return (0.40 * d.get("hubness", 0) + 0.25 * d.get("tier_w", 0)
                    + 0.20 * d.get("freshness", 0) + 0.15 * d.get("status_w", 0))
        seeds = sorted(G, key=support_score, reverse=True)
    support = seeds[:4]
    top = support[0]
    # One grounded claim per line, each carrying its citation, so claim-level
    # citation integrity is measured cleanly.
    lines = [
        f"The most corroborated, highest-trust reference for this query is [{top}] "
        f"(in-degree {G.nodes[top].get('in_degree', 0)} in the retrieved subgraph): "
        f"{G.nodes[top]['title']}."
    ]
    others = support[1:]
    if others:
        lines.append("It is corroborated within the subgraph by "
                     + ", ".join(f"[{n}]" for n in others) + ".")
    for n in support:
        d = G.nodes[n]
        clause = " ".join(d["body"].split()).split(". ")[0][:160].rstrip(".")
        lines.append(f"[{n}] ({d['type']}, {d['status']}): {clause}.")

    gaps = _structural_gaps(G)
    return _finalize(G, "\n".join(lines), support, gaps, method="extractive")


def _synthesize_extractive_causal(G: nx.DiGraph, query: str, causal_path) -> DecisionResult:
    """Narrate the traced causal path deterministically, citing per-step evidence.

    Every line carries bracketed [ID] citations for the step's nodes and its
    evidence source, so the narration is grounded in — and only in — the path
    the traversal already found. Uncited free-association is structurally
    impossible here: the lines ARE the traced edges.
    """
    entry = causal_path.entry
    rc = ", ".join(f"[{r}]" for r in causal_path.root_causes) or "the traced origin"
    lines = [
        f"This was assessed by tracing the causal chain from the impact [{entry}] "
        f"back through the knowledge graph to its root cause(s): {rc}."
    ]
    for i, s in enumerate(causal_path.steps, 1):
        if s.evidence:
            ev = f' Evidence [{s.evidence[0]}]: "{s.evidence[1]}"'
        else:
            ev = " Evidence: MISSING for this step (a gap)."
        lines.append(
            f"{i}. [{s.src}] --{s.rel}--> [{s.target}] "
            f"({s.target_type}: {s.target_title}).{ev}"
        )
    # Persona accountability (who raised the risk / made the call / owned the code).
    role_phrase = {
        "raised_risk": "the risk was raised by",
        "made_the_call": "the decision was made by",
        "owned_the_code": "the dropped safeguard was owned by",
    }
    if causal_path.personas:
        acc = "; ".join(
            f"{role_phrase.get(role, role)} [{nid}]"
            for role, nid in causal_path.personas.items()
        )
        lines.append(f"Accountability: {acc}.")

    cited = [n for n in causal_path.context_ids() if n in G.nodes]
    gaps = _structural_gaps(G)
    gaps += [f"Causal step [{s.src}]--{s.rel}-->[{s.target}] has no evidence."
             for s in causal_path.unevidenced]
    return _finalize(G, "\n".join(lines), cited, gaps, method="extractive-causal")


def _structural_gaps(G: nx.DiGraph) -> list[str]:
    gaps: list[str] = []
    for a, b in G.graph.get("contradictions", []):
        gaps.append(f"References {a} and {b} contradict each other.")
    for src, rel, tgt in G.graph.get("dangling", []):
        gaps.append(f"{src} references {tgt} ({rel}) which is not available in the corpus.")
    stale = [n for n in G if G.nodes[n].get("status") == "deprecated"]
    if stale:
        gaps.append(f"Some references are deprecated/superseded: {', '.join(sorted(stale))}.")
    return gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _finalize(G, decision_text, cited, gaps, *, method) -> DecisionResult:
    """Validate citations against the subgraph; flag any hallucinated IDs."""
    in_graph = set(G.nodes)
    seen: list[str] = []
    halluc: list[str] = []
    for c in cited:
        if c in in_graph and c not in seen:
            seen.append(c)
        elif c not in in_graph and c not in halluc:
            halluc.append(c)
    # Also catch IDs mentioned inline but not declared as cited.
    for c in NODE_ID_RE.findall(decision_text):
        if c in in_graph and c not in seen:
            seen.append(c)
        elif c not in in_graph and c not in halluc:
            halluc.append(c)
    return DecisionResult(
        decision_text=decision_text,
        cited_node_ids=seen,
        model_noted_gaps=list(gaps),
        method=method,
        hallucinated_citations=halluc,
    )


def decide_for_query(
    corpus: Corpus, query: str, *, retriever=None, n_seeds: int = 8
) -> tuple[nx.DiGraph, DecisionResult]:
    """Convenience: retrieve (graph-boosted) -> build context map -> synthesize.

    Uses the graph-boosted retriever (our best per Phase 3) by default so the
    seeds are both query-relevant and structurally strong; pass ``retriever`` to
    override (e.g. an access-filtered retriever in Phase 6).
    """
    if retriever is None:
        from .graph_builder import GraphBoostedRetriever
        retriever = GraphBoostedRetriever(build_retrievers(corpus)["hybrid"], corpus)
    top = [nid for nid, _ in retriever.retrieve(query, k=n_seeds)]
    G = build_context_map(corpus, top, query=query)
    return G, synthesize_decision(G, query)


if __name__ == "__main__":
    corpus = Corpus.load()
    q = "what is the root cause of the SG settlement batch failures?"
    G, result = decide_for_query(corpus, q)
    print(f"query: {q}")
    print(f"method: {result.method}\n")
    print("DECISION:")
    print(result.decision_text)
    print(f"\ncited node ids: {result.cited_node_ids}")
    print(f"model-noted gaps: {result.model_noted_gaps}")
    if result.hallucinated_citations:
        print(f"!! hallucinated citations (not in subgraph): {result.hallucinated_citations}")
    else:
        print("citation check: all citations reference real subgraph nodes.")
