# Graph-Informed Retrieval & Confidence Engine (PS-019 PoC)

> Not a RAG chatbot. The interesting problem is the **read path**: given a
> critical issue, retrieve all interrelated references, build a context map
> (graph) from them, reason over the *structure* of that map to produce a
> decision, and emit a **confidence score derived from observable features of
> the retrieved subgraph** — not from an LLM self-reporting a number.

**Central thesis (the thing we validate with numbers):** the structure of the
retrieved subgraph predicts the confidence *and* correctness of the decision. A
decision backed by a densely corroborated, fresh, contradiction-free subgraph is
more trustworthy than one resting on a sparse, stale, or self-contradicting one.

## Scenario

A single critical incident — *"production payment-settlement batch failing
intermittently in the Singapore region"* — resolving into a rich, labeled
context map of ~30–50 interrelated nodes (Jira-style tickets, comments,
resolutions/postmortems, runbooks, docs).

## Architecture (by phase)

| Phase | Module | What it does |
|------|--------|--------------|
| 1 | `src/corpus.py` | Load + validate the synthetic corpus; report gaps |
| 2 | `src/retrieval.py`, `src/eval_retrieval.py` | Keyword + embedding + hybrid (RRF); P@k/R@k |
| 3 | `src/graph_builder.py` | Build context map; per-node structural signals; graph-lift |
| 4 | `src/decision.py` | LLM synthesis over the subgraph (the `think` stage) |
| 5 | `src/confidence.py` | Transparent, feature-based confidence + gap report |
| 6 | `src/access.py` | Role-based security-label filtering at the retrieval boundary |
| 7 | `src/eval_confidence.py` | Calibration curve + ECE + thesis validation |
| 8 | `app/streamlit_app.py` | Demo UI |

Design is deliberately **narrow and deep**: in-memory `networkx` (a ~40-node
graph traverses in microseconds — Neo4j would be ceremony at this scale),
in-memory access control, synthetic ground-truth corpus so every metric is
measurable. Neo4j, multi-issue scale, and learned calibration are explicitly
deferred to a final phase.

## Setup

```bash
pip install -r requirements.txt
# decision synthesis (Phase 4+) reads the API key from the environment:
export ANTHROPIC_API_KEY=...      # never hardcoded
```

### Embedding backend note

The intended embedder is `sentence-transformers/all-MiniLM-L6-v2`. The retrieval
layer loads it automatically whenever the weights are reachable (HuggingFace
egress, or a local dir via `KGCE_ST_MODEL`). In a sandbox where HuggingFace is
network-blocked it falls back to a **deterministic, network-free dense embedder**
(hashed char+word n-gram TF-IDF, cosine similarity). The retriever interface and
the RRF fusion are identical either way, so MiniLM is a zero-code swap. Force a
backend with `KGCE_EMBED_BACKEND=st|hashing|auto`. *The retrieval numbers below
were produced with the fallback embedder; expect them to shift (typically hybrid
improves) once MiniLM weights are available.*

## Run

```bash
python -m src.corpus            # corpus stats + seeded-condition checklist
python -m src.retrieval         # sample retrieval for one query
python -m src.eval_retrieval    # P@5 / R@5 table
```

## Results

> Running log of measured numbers. Every quality claim here is produced by an
> eval harness, not asserted.

### Retrieval (Phase 2) — mean over 10 labeled eval queries

Embedding backend: `hashing-tfidf` (MiniLM fallback — see note above).

| Retriever | P@5 | R@5 |
|-----------|-----|-----|
| keyword (BM25) | 0.340 | 0.500 |
| embedding (hashing-tfidf) | 0.380 | 0.550 |
| hybrid (RRF, k=60) | 0.360 | 0.517 |

Hybrid is competitive with the better single retriever and beats keyword.
Note on P@5: several eval queries have fewer than 5 truly-relevant nodes, so
P@5 is structurally capped below 1.0 for those; **R@5 is the more informative
metric** for this corpus.

### Graph lift (Phase 3) — re-rank by subgraph hub-ness

The retrieved set is assembled into a context map (in-memory `networkx.DiGraph`
with typed edges); each candidate is re-ranked by its **hub-ness** (in-degree
within the subgraph — how many other retrieved references point at it). Measured
against the plain hybrid baseline (pool=12, α=0.6, mean over the eval set):

| Retriever | P@5 | R@5 |
|-----------|-----|-----|
| hybrid (RRF) | 0.360 | 0.517 |
| **hybrid + graph-boost** | **0.460** | **0.707** |
| **lift** | **+0.100** | **+0.190** |

This is the empirical version of GBrain's graph-over-vector claim: a real,
measured improvement from using the structure of the retrieved subgraph. The top
hub for the root-cause query is the postmortem `RES-12` (in-degree 7), i.e. the
node the most other references corroborate.

### Confidence model (Phase 5) — transparent, decomposable

Confidence is a weighted combination of six observable subgraph features, each
normalized to [0,1]; **the score is exactly the (clamped) sum of contributions**,
so it reconstructs by hand from the breakdown. Example (root-cause query):

| feature | value | weight | contribution |
|---|---|---|---|
| corroboration | 1.00 | +0.35 | +0.350 |
| source_tier | 0.65 | +0.20 | +0.130 |
| freshness | 0.85 | +0.25 | +0.213 |
| citation_integrity | 1.00 | +0.20 | +0.200 |
| contradiction | 0.00 | −0.30 | −0.000 |
| coverage | 0.56 | −0.15 | −0.084 |
| **score** | | | **0.809 → HIGH** |

The additive subtotal is then multiplied by an **evidential-sufficiency gate**
(`min(1, connected_nodes/6)`): a tiny subgraph cannot yield high confidence
however fresh/high-tier its few nodes are. `score = subtotal × sufficiency`, still
fully reconstructable. Plus a structured **gap report** (contradictions, dangling
references, stale nodes, uncited claims) — the "what we don't know" surface.

**Band alignment on the eval set (default role, extractive decision):** 6/10
predicted bands match the labels; the HIGH band is identified cleanly (high
mean 0.72 vs medium/low ≈ 0.53). The misses are understood: Q4/Q6/Q8 are
borderline medium↔low calls, and **Q9 is role-dependent** — its supporting node
is `restricted`, so at the default (lead) role it is visible and the query is
answerable (not low); under an `intern` role it drops to low (shown in Phase 6).
Note: `citation_integrity` is degenerate (≈1.0) under the extractive fallback,
so discrimination here comes from the structural features; with real LLM
synthesis it also penalizes uncited/hallucinated claims.

### Access control (Phase 6) — filtering as sub-graph removal

Roles map to security-label clearances (`intern → {public}`,
`engineer → {public, internal}`, `lead → +restricted`, `hr → +hr_only`).
Filtering happens at the **corpus/retrieval boundary**, so a forbidden node never
enters retrieval, the context map, the LLM prompt, or the confidence score — and
a link to a now-invisible node becomes a dangling gap for that role. Enforcement
is in **code, not in the prompt**: prompt-level access control is trivially
prompt-injectable, and the reference model requires that the query engine
literally cannot see the node.

Same query, different role (measured):

| Query | intern | engineer | lead |
|---|---|---|---|
| *root cause?* | 0.222 (LOW) | 0.590 (MED) | 0.670 (HIGH) |
| *customer SLA-credit impact?* | 0.222 (LOW) | 0.638 (HIGH) | 0.639 (HIGH) |

The intern (only 2 public docs visible) is correctly LOW. For the root-cause
query, the **lead is HIGH while the engineer is only MEDIUM** — because the
restricted vendor RCA `RES-13` corroborates the root cause and only the lead can
see it. This is the access requirement and a confidence-from-structure result in
one demo beat.

*(Calibration/ECE and thesis-validation results land in Phase 7.)*

## Out of scope (named deliberately)

Real ingestion/entity-extraction, OAuth/AD federation, audit logging, embedding
infrastructure at scale, and Neo4j — until a final phase, and only when scale
justifies it. Access control is enforced **in code at the retrieval boundary**,
never via prompt instructions (LLM-layer access control is trivially
prompt-injectable).
