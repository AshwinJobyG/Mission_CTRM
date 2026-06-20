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
metric** for this corpus. The graph-adjacency re-rank (Phase 3) is measured as a
lift *on top of* this hybrid baseline.

*(Graph-lift, calibration/ECE, and thesis-validation results land in Phases 3
and 7.)*

## Out of scope (named deliberately)

Real ingestion/entity-extraction, OAuth/AD federation, audit logging, embedding
infrastructure at scale, and Neo4j — until a final phase, and only when scale
justifies it. Access control is enforced **in code at the retrieval boundary**,
never via prompt instructions (LLM-layer access control is trivially
prompt-injectable).
