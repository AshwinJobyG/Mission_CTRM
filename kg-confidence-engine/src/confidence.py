"""Feature-based confidence model — the centerpiece.

Confidence is a transparent, weighted combination of observable features of the
retrieved subgraph. It is NOT an LLM self-report. The single hard rule: the score
must be reconstructable by hand from the breakdown — every feature exposes its
normalized value [0,1], its (signed) weight, and its contribution, and the score
is exactly the (clamped) sum of contributions.

Features (4 positive, weights sum to 1; 2 penalties, negative weights):
  + corroboration     how many cited supports are corroborated hubs
  + source_tier       average trust tier of the cited supports
  + freshness         how recent the cited supports are
  + citation_integrity fraction of decision claims that carry a citation
  - contradiction     contradicts-pairs touching the cited/retrieved set
  - coverage          dangling/unresolved references in the cited cluster

Phase D adds a 7th feature, ``path_completeness``, scoped by query type:
  + path_completeness fraction of the traced causal chain that is present AND
                      evidenced — carries weight ONLY for causal queries; its
                      weight is 0 for factual queries so existing calibration is
                      byte-for-byte unchanged.

Output: ConfidenceResult{score, band, breakdown, gap_report}.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import networkx as nx

from .decision import DecisionResult, NODE_ID_RE

# Signed weights for FACTUAL queries. Positive features sum to 1.0; penalties are
# negative; path_completeness weight is 0 so factual scores are identical to the
# pre-integration model (regression-locked).
WEIGHTS = {
    "corroboration": 0.35,
    "source_tier": 0.20,
    "freshness": 0.25,
    "citation_integrity": 0.20,
    "contradiction": -0.30,
    "coverage": -0.15,
    "path_completeness": 0.0,
}

# Signed weights for CAUSAL queries. The four positive structural features are
# scaled to 0.70 and path_completeness takes the remaining 0.30, so the positive
# weights (incl. path_completeness) still sum to 1.0; penalties are unchanged.
CAUSAL_WEIGHTS = {
    "corroboration": 0.245,
    "source_tier": 0.14,
    "freshness": 0.175,
    "citation_integrity": 0.14,
    "contradiction": -0.30,
    "coverage": -0.15,
    "path_completeness": 0.30,
}

# Bracketed citation detector used ONLY for causal narration scoring (the
# incident corpus uses ids like NGPOWER-145 / PERSON-anupam that the factual
# NODE_ID_RE intentionally does not match). Factual scoring is untouched.
BRACKET_ID_RE = re.compile(r"\[([A-Za-z][\w.-]*)\]")

BAND_THRESHOLDS = {"high": 0.62, "medium": 0.48}  # else "low"


@dataclass
class ConfidenceResult:
    score: float
    band: str
    breakdown: dict[str, dict]  # feature -> {value, weight, contribution}
    subtotal: float = 0.0       # sum of feature contributions (pre-gate)
    sufficiency: float = 1.0    # evidential-sufficiency gate in [0,1]
    gap_report: list[dict] = field(default_factory=list)

    def reconstruct(self) -> float:
        """score = (sum of contributions) x sufficiency — verifies reproducibility."""
        return sum(f["contribution"] for f in self.breakdown.values()) * self.sufficiency


def band_for(score: float) -> str:
    if score >= BAND_THRESHOLDS["high"]:
        return "high"
    if score >= BAND_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def score_confidence(
    G: nx.DiGraph, decision: DecisionResult, *,
    causal_path=None, query_type: str = "factual",
) -> ConfidenceResult:
    """Score the decision over the subgraph.

    ``query_type`` selects the weight vector: ``"factual"`` uses :data:`WEIGHTS`
    (path_completeness weight 0 ⇒ score identical to the pre-integration model);
    ``"causal"`` uses :data:`CAUSAL_WEIGHTS` and scores ``path_completeness`` over
    the traced ``causal_path``. Called with no kwargs, behaviour is unchanged.
    """
    causal = query_type == "causal"
    weights = CAUSAL_WEIGHTS if causal else WEIGHTS
    cited = [c for c in decision.cited_node_ids if c in G]
    values = {
        "corroboration": _corroboration(G, cited),
        "source_tier": _avg(G, cited, "tier_w"),
        "freshness": _avg(G, cited, "freshness"),
        "citation_integrity": _citation_integrity(G, decision, causal=causal),
        "contradiction": _contradiction(G, cited),
        "coverage": _coverage(G, cited),
        "path_completeness": _path_completeness(causal_path),
    }
    breakdown: dict[str, dict] = {}
    for feat, val in values.items():
        w = weights[feat]
        breakdown[feat] = {
            "value": round(val, 4),
            "weight": w,
            "contribution": round(val * w, 4),
        }
    subtotal = sum(f["contribution"] for f in breakdown.values())
    # Evidential-sufficiency gate: confidence is capped when too few independent
    # corroborating supports exist, regardless of how fresh/high-tier they are.
    # This encodes the thesis (sparse support => not trustworthy) and fixes the
    # perverse "tiny but healthy subgraph => high confidence" case (e.g. an
    # access-restricted role left with only a couple of generic docs).
    n_connected = sum(1 for n in G if G.degree(n) >= 1)
    sufficiency = min(1.0, n_connected / 6.0)
    score = max(0.0, min(1.0, subtotal * sufficiency))
    return ConfidenceResult(
        score=round(score, 4),
        band=band_for(score),
        breakdown=breakdown,
        subtotal=round(subtotal, 4),
        sufficiency=round(sufficiency, 4),
        gap_report=_gap_report(G, decision, causal_path=causal_path),
    )


# ---- feature computations ---------------------------------------------------

def _avg(G, cited, attr) -> float:
    if not cited:
        return 0.0
    return sum(G.nodes[c].get(attr, 0.0) for c in cited) / len(cited)


def _corroboration(G, cited) -> float:
    """Count cited supports that are corroborated (in-degree >= 1), saturating."""
    if not cited:
        return 0.0
    corroborated = sum(1 for c in cited if G.nodes[c].get("in_degree", 0) >= 1)
    return min(1.0, corroborated / 4.0)


def _unit_is_cited(unit: str, G, causal: bool) -> bool:
    """Whether a claim unit carries a citation.

    Factual: the original ``NODE_ID_RE`` test (unchanged). Causal: a bracketed
    [ID] referring to a node present in the subgraph — the incident corpus uses
    id forms (NGPOWER-145, PERSON-anupam) that NODE_ID_RE deliberately ignores.
    """
    if causal:
        return any(tok in G for tok in BRACKET_ID_RE.findall(unit))
    return bool(NODE_ID_RE.search(unit))


def _citation_integrity(G, decision: DecisionResult, *, causal: bool = False) -> float:
    """Fraction of decision claims that carry a citation, scaled by hallucination."""
    units = [u.strip() for u in re.split(r"[\n.]", decision.decision_text) if len(u.strip()) > 20]
    if not units:
        base = 1.0
    else:
        cited_units = sum(1 for u in units if _unit_is_cited(u, G, causal))
        base = cited_units / len(units)
    n_valid = len([c for c in decision.cited_node_ids if c in G])
    n_halluc = len(decision.hallucinated_citations)
    halluc_factor = n_valid / (n_valid + n_halluc) if (n_valid + n_halluc) else 1.0
    return base * halluc_factor


def _path_completeness(causal_path) -> float:
    """Fraction of the traced causal chain that is present AND evidenced.

    Combines three structural sub-signals (all in [0,1]):
      * evidence coverage — fraction of traced steps carrying evidence,
      * persona accountability — fraction of {risk-raiser, decision-maker,
        code-owner} the path identifies,
      * root-cause reached — whether the chain terminates at a Risk root cause.
    Returns 0.0 when there is no path (e.g. a factual query, where the feature
    weight is 0 anyway).
    """
    if not causal_path or not getattr(causal_path, "steps", None):
        return 0.0
    n_steps = len(causal_path.steps)
    step_ev = causal_path.evidenced_steps / n_steps if n_steps else 0.0
    persona = len(causal_path.personas) / 3.0
    root = 1.0 if causal_path.root_causes else 0.0
    return max(0.0, min(1.0, 0.7 * step_ev + 0.2 * persona + 0.1 * root))


def _contradiction(G, cited) -> float:
    """Penalty value: contradicts-pairs, weighted higher when touching cited nodes.

    A contradiction directly in the cited support is severe (the decision rests on
    evidence that other retrieved references dispute)."""
    cited_set = set(cited)
    touching = nontouching = 0
    for a, b in G.graph.get("contradictions", []):
        if a in cited_set or b in cited_set:
            touching += 1
        else:
            nontouching += 1
    return min(1.0, touching * 1.0 + 0.4 * nontouching)


def _coverage(G, cited) -> float:
    """Penalty value: dangling references, weighted higher from cited nodes."""
    cited_set = set(cited)
    touching = nontouching = 0
    for src, _rel, _tgt in G.graph.get("dangling", []):
        if src in cited_set:
            touching += 1
        else:
            nontouching += 1
    return min(1.0, (touching + 0.4 * nontouching) / 2.5)


# ---- gap report (structured "what we don't know") ---------------------------

def _gap_report(G, decision: DecisionResult, *, causal_path=None) -> list[dict]:
    gaps: list[dict] = []
    # Causal-specific gaps named first (the most actionable for a "why" answer):
    # a step whose edge has no evidence, a decision with no rationale, a risk with
    # no owner. These point at the exact missing/unevidenced link.
    if causal_path is not None:
        gaps.extend(_causal_gaps(causal_path))
    for a, b in G.graph.get("contradictions", []):
        gaps.append({"type": "contradiction", "detail": f"{a} contradicts {b}", "nodes": [a, b]})
    for src, rel, tgt in G.graph.get("dangling", []):
        gaps.append({"type": "dangling_reference",
                     "detail": f"{src} --{rel}--> {tgt} (target missing from corpus)",
                     "nodes": [src]})
    for n in sorted(G):
        if G.nodes[n].get("status") == "deprecated":
            gaps.append({"type": "stale", "detail": f"{n} is deprecated/superseded", "nodes": [n]})
    # uncited factual claims in the decision
    units = [u.strip() for u in re.split(r"[\n.]", decision.decision_text) if len(u.strip()) > 30]
    uncited = [u for u in units if not NODE_ID_RE.search(u)]
    for u in uncited:
        gaps.append({"type": "uncited_claim", "detail": u[:120], "nodes": []})
    if decision.hallucinated_citations:
        gaps.append({"type": "hallucinated_citation",
                     "detail": "cited IDs not in subgraph: " + ", ".join(decision.hallucinated_citations),
                     "nodes": decision.hallucinated_citations})
    return gaps


def _causal_gaps(causal_path) -> list[dict]:
    """Name the specific missing / unevidenced links in the traced causal chain."""
    gaps: list[dict] = []
    if not getattr(causal_path, "steps", None):
        return gaps
    # A causal step whose edge carries no evidence.
    for s in causal_path.unevidenced:
        gaps.append({
            "type": "unevidenced_causal_link",
            "detail": f"Causal link [{s.src}] --{s.rel}--> [{s.target}] "
                      f"has no recorded evidence.",
            "nodes": [s.src, s.target],
        })
    # A decision on the path with no recorded rationale (no outgoing causal step).
    decided = {s.src for s in causal_path.steps if s.target_type == "decision"}
    decided |= {s.target for s in causal_path.steps if s.target_type == "decision"}
    srcs = {s.src for s in causal_path.steps}
    for d in sorted(decided):
        if d not in srcs:
            gaps.append({
                "type": "decision_without_rationale",
                "detail": f"Decision [{d}] has no recorded rationale node "
                          f"(no onward causal link to a risk/meeting).",
                "nodes": [d],
            })
    # A risk root cause with no identified owner/raiser.
    if causal_path.root_causes and "raised_risk" not in causal_path.personas:
        gaps.append({
            "type": "risk_without_owner",
            "detail": f"Risk root cause {causal_path.root_causes} has no recorded "
                      f"owner (no RAISED_RISK edge to a person).",
            "nodes": list(causal_path.root_causes),
        })
    return gaps


if __name__ == "__main__":
    from .corpus import Corpus
    from .decision import decide_for_query

    corpus = Corpus.load()
    q = "what is the root cause of the SG settlement batch failures?"
    G, decision = decide_for_query(corpus, q)
    conf = score_confidence(G, decision)

    print(f"query: {q}")
    print(f"\nCONFIDENCE: {conf.score:.3f}  ->  band: {conf.band.upper()}\n")
    print(f"{'feature':<20} {'value':>8} {'weight':>8} {'contrib':>9}")
    print("-" * 48)
    for feat, d in conf.breakdown.items():
        print(f"{feat:<20} {d['value']:>8.3f} {d['weight']:>+8.2f} {d['contribution']:>+9.3f}")
    print("-" * 48)
    print(f"{'subtotal':<20} {'':>8} {'':>8} {conf.subtotal:>+9.3f}")
    print(f"{'x sufficiency gate':<20} {'':>8} {'':>8} {conf.sufficiency:>9.3f}")
    print(f"{'= score':<20} {'':>8} {'':>8} {conf.score:>9.3f}")
    print(f"(reconstruct check: {conf.reconstruct():.3f})")
    print("\nGAP REPORT:")
    for g in conf.gap_report:
        print(f"  [{g['type']}] {g['detail']}")
