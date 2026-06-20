"""Demo UI — walks the whole thesis on one screen.

ask -> retrieved set (search) -> context map (graph) -> grounded decision (think)
-> confidence breakdown + gap report, with a role selector that visibly changes
the result. A second tab surfaces the Phase 7 rigor (calibration + thesis).

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.access import ROLE_CLEARANCES, can_see, filtered_corpus  # noqa: E402
from src.causal_traversal import answer_query, classify_query  # noqa: E402
from src.confidence import WEIGHTS, score_confidence  # noqa: E402
from src.corpus import CORPUS_PATHS, Corpus  # noqa: E402
from src.decision import NODE_ID_RE, synthesize_decision  # noqa: E402
from src.graph_builder import GraphBoostedRetriever, build_context_map  # noqa: E402
from src.retrieval import build_retrievers  # noqa: E402
from src.schema import node_type  # noqa: E402

DATA_DIR = ROOT / "data"

# colour nodes by source tier
TIER_COLOR = {
    "resolution": "#2ca02c", "runbook": "#17becf", "doc": "#1f77b4",
    "incident": "#d62728", "ticket": "#ff7f0e", "comment": "#7f7f7f",
}
CAUSAL_EDGE_COLOR = "#9467bd"   # distinct colour for traced causal-path edges


# ---- pure helpers (importable / testable without streamlit) -----------------

def context_map_html(G, cited: set[str], path_edges=frozenset(),
                     path_nodes=frozenset(), height: str = "520px") -> str:
    """Render the context map as an interactive pyvis HTML string.

    ``path_edges`` (set of (u, v)) are the traced causal-path edges and are
    drawn distinctly (thick purple) from the rest of the subgraph; ``path_nodes``
    get a purple halo so the spine reads at a glance.
    """
    from pyvis.network import Network

    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#222222")
    net.barnes_hut(spring_length=120)
    for n in G:
        d = G.nodes[n]
        tier = d.get("source_tier", "ticket")
        if n in cited:
            border = "#111111"
        elif n in path_nodes:
            border = CAUSAL_EDGE_COLOR
        else:
            border = TIER_COLOR.get(tier, "#999")
        size = 14 + 22 * d.get("hubness", 0.0)
        nt = d.get("node_type", "record")
        title = (f"{d['title']}<br>node_type={nt} | type={d.get('type','-')} | "
                 f"status={d.get('status','-')} | tier={tier} | {d.get('date','-')}<br>"
                 f"in-degree={d.get('in_degree',0)} hubness={d.get('hubness',0):.2f} "
                 f"fresh={d.get('freshness',0):.2f}")
        bw = 4 if (n in cited or n in path_nodes) else 1
        net.add_node(n, label=n, title=title, color={
            "background": TIER_COLOR.get(tier, "#999"),
            "border": border}, borderWidth=bw, size=size)
    for u, v, data in G.edges(data=True):
        rel = data.get("rel", "")
        on_path = (u, v) in path_edges
        if on_path:
            color, width = CAUSAL_EDGE_COLOR, 4
        elif rel == "contradicts":
            color, width = "#d62728", 1
        else:
            color, width = "#aaaaaa", 1
        net.add_edge(u, v, label=rel, color=color, arrows="to", width=width)
    return net.generate_html(notebook=False)


def causal_steps_rows(causal_path) -> list[dict]:
    """Flatten a CausalPath into display rows (pure; testable without streamlit)."""
    rows = []
    for i, s in enumerate(causal_path.steps, 1):
        rows.append({
            "step": i,
            "from": s.src,
            "rel": s.rel,
            "to": s.target,
            "to_type": s.target_type,
            "evidence_source": s.evidence[0] if s.evidence else "(missing)",
            "evidence": s.evidence[1] if s.evidence else "— no evidence (gap) —",
        })
    return rows


PERSONA_LABEL = {
    "raised_risk": "🚩 raised the risk",
    "made_the_call": "🧭 made the call",
    "owned_the_code": "🛠️ owned the code",
    "customer_impacted": "💥 impacted customer",
}


def breakdown_figure(conf):
    """Signed-contribution bar chart of the confidence breakdown."""
    feats = list(conf.breakdown.keys())
    contribs = [conf.breakdown[f]["contribution"] for f in feats]
    colors = ["#2ca02c" if c >= 0 else "#d62728" for c in contribs]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh(feats, contribs, color=colors, alpha=0.85)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("contribution to score")
    ax.set_title(f"subtotal {conf.subtotal:+.3f}  ×  sufficiency {conf.sufficiency:.2f}"
                 f"  =  {conf.score:.3f}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def render_decision_md(text: str) -> str:
    """Bold the inline [ID] citations for readability."""
    return NODE_ID_RE.sub(lambda m: f"**`{m.group(0)}`**", text)


# ---- streamlit app ----------------------------------------------------------

def main() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="Graph-Informed Confidence Engine", layout="wide")

    @st.cache_resource
    def _corpus(name: str):
        return Corpus.load_named(name)

    @st.cache_resource
    def _retriever(name: str, role: str):
        view = filtered_corpus(_corpus(name), role)
        return view, GraphBoostedRetriever(build_retrievers(view)["hybrid"], view)

    st.title("Graph-Informed Retrieval & Confidence Engine")
    st.caption("PS-019 PoC — retrieve interrelated references, build a context "
               "map, reason over its structure, and emit a confidence score "
               "derived from the subgraph (not self-reported by the LLM). "
               "Stage 3.5 traces the causal path for 'why' questions.")

    tab_demo, tab_rigor = st.tabs(["🔎 Demo", "📊 Results (rigor)"])

    with tab_demo:
        c1, c2 = st.columns([3, 1])
        corpus_name = c2.selectbox(
            "Corpus", list(CORPUS_PATHS),
            format_func=lambda k: {"default": "SG settlement (original)",
                                   "incident": "NGPOWER P0 (causal)"}.get(k, k))
        default_q = ("Why did NGPOWER-145 become a P0 incident?"
                     if corpus_name == "incident"
                     else "what is the root cause of the SG settlement batch failures?")
        query = c1.text_input("Critical issue / question", default_q)
        role = c2.selectbox("Role", list(ROLE_CLEARANCES), index=2)
        n_seeds = c2.slider("retrieved (N)", 4, 12, 8)

        corpus = _corpus(corpus_name)
        view, retriever = _retriever(corpus_name, role)

        # Unified pipeline: routes the query, inserts Stage 3.5 when causal.
        ans = answer_query(corpus, query, role=role, retriever=retriever, n_seeds=n_seeds)
        G, decision, conf = ans.graph, ans.decision, ans.confidence
        causal_path = ans.causal_path
        cited = set(decision.cited_node_ids)
        ranked = retriever.retrieve(query, k=n_seeds)

        # Query-type indicator
        if ans.query_type == "causal":
            st.success("🧭 **Causal** question — Stage 3.5 traced the root-cause→"
                       "impact path before the LLM was called; the answer narrates "
                       "that path and cites the evidence on each step.")
        else:
            st.info("🔎 **Factual** question — standard retrieve → reason path "
                    "(no causal traversal).")

        hidden = [n["id"] for n in corpus if not can_see(n, role)]
        st.caption(f"Role **{role}** can see {len(view)}/{len(corpus)} nodes. "
                   f"Hidden: {len(hidden)} — enforced in code at the retrieval boundary.")

        # Stage 1 — retrieved references (the search stage)
        st.subheader("Stage 1 · Retrieved references (search)")
        st.table([{"rank": i + 1, "id": nid, "score": round(s, 4),
                   "node_type": node_type(view.nodes[nid]),
                   "title": view.nodes[nid]["title"]}
                  for i, (nid, s) in enumerate(ranked)])

        # Stage 2 — context map (the wow moment)
        path_edges = frozenset(causal_path.edges_uv()) if causal_path else frozenset()
        path_nodes = frozenset(causal_path.nodes()) if causal_path else frozenset()
        st.subheader("Stage 2 · Context map (everything is interrelated)")
        st.caption("Node size = hub-ness; colour = source tier; thick black border "
                   "= cited; red edges = contradictions; **purple edges/nodes = the "
                   "traced causal path**.")
        components.html(context_map_html(G, cited, path_edges, path_nodes),
                        height=540, scrolling=True)

        # Stage 3.5 — causal path panel (only for causal questions)
        if causal_path:
            st.subheader("Stage 3.5 · Causal path (the traversal did the reasoning)")
            st.caption("Walked backward from the impact over causal edges. The LLM "
                       "narrates THIS chain — it does not free-associate.")
            if causal_path.personas:
                cols = st.columns(len(causal_path.personas))
                for col, (role_key, nid) in zip(cols, causal_path.personas.items()):
                    col.metric(PERSONA_LABEL.get(role_key, role_key), nid)
            for r in causal_steps_rows(causal_path):
                st.markdown(
                    f"**{r['step']}.** `{r['from']}` —**{r['rel']}**→ `{r['to']}` "
                    f"*({r['to_type']})*  \n"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;📎 *evidence* `{r['evidence_source']}`: "
                    f"“{r['evidence']}”")

        # Stage 3 — decision (the think stage)
        st.subheader("Stage 3 · Decision (think)")
        st.markdown(render_decision_md(decision.decision_text))
        st.caption(f"synthesis: {decision.method} · cited: "
                   + ", ".join(f"`{c}`" for c in decision.cited_node_ids))
        if decision.hallucinated_citations:
            st.error("Hallucinated citations (not in subgraph): "
                     + ", ".join(decision.hallucinated_citations))

        # Stage 4 — confidence
        st.subheader("Stage 4 · Confidence (from graph structure)")
        m1, m2 = st.columns([1, 2])
        m1.metric("confidence", f"{conf.score:.3f}", conf.band.upper())
        if ans.query_type == "causal":
            pc = conf.breakdown["path_completeness"]
            m1.metric("path completeness", f"{pc['value']:.2f}",
                      f"weight {pc['weight']:.2f}")
        m2.pyplot(breakdown_figure(conf))
        with st.expander("Gap report — what we don't know", expanded=True):
            if not conf.gap_report:
                st.write("No structural gaps detected.")
            for g in conf.gap_report:
                st.write(f"- **{g['type']}** — {g['detail']}")

    with tab_rigor:
        st.subheader("Calibration (reliability diagram)")
        if (DATA_DIR / "calibration.png").exists():
            st.image(str(DATA_DIR / "calibration.png"), width=480)
        st.subheader("Thesis validation — structure predicts correctness")
        if (DATA_DIR / "thesis_validation.png").exists():
            st.image(str(DATA_DIR / "thesis_validation.png"), width=720)
        tv_path = DATA_DIR / "thesis_validation.json"
        if tv_path.exists():
            tv = json.loads(tv_path.read_text())["thesis"]
            st.write(f"**confidence ↔ correctness:** Pearson r = "
                     f"{tv['pearson_r_with_correctness']['confidence']:+.3f} "
                     f"over {tv['n_conditions']} conditions")
            st.write(f"**dense vs sparse accuracy:** "
                     f"{tv['rich_subgraph_accuracy']:.2f} vs "
                     f"{tv['poor_subgraph_accuracy']:.2f}")
            st.json(tv["pearson_r_with_correctness"])
        rt_path = DATA_DIR / "retrieval_eval.json"
        if rt_path.exists():
            st.subheader("Retrieval P@5 / R@5")
            res = json.loads(rt_path.read_text())["results"]
            st.table([{"retriever": v["name"], "P@5": v["p_at_k"], "R@5": v["r_at_k"]}
                      for v in res.values()])

        # --- Causal capability (Stage 3.5) ---
        ce_path = DATA_DIR / "causal_eval.json"
        if ce_path.exists():
            ce = json.loads(ce_path.read_text())
            st.markdown("---")
            st.subheader("Causal capability — Stage 3.5 (incident corpus)")
            pr = ce["path_retrieval"]
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("path-retrieval precision", f"{pr['mean_precision']:.2f}")
            cc2.metric("path-retrieval recall", f"{pr['mean_recall']:.2f}")
            cc3.metric("causal ECE", f"{ce['calibration']['ece']:.3f}")
            t = ce["thesis_causal"]
            st.write(f"**path-completeness ↔ correctness:** Pearson r = "
                     f"{t['pearson_path_completeness_vs_correct']:+.3f} over "
                     f"{t['n_conditions']} conditions "
                     f"(accuracy {t['overall_accuracy']:.2f}).")
            if (DATA_DIR / "causal_calibration.png").exists():
                st.image(str(DATA_DIR / "causal_calibration.png"), width=480)

        st.caption("Regenerate: `python -m src.eval_confidence full` (factual) and "
                   "`python -m src.eval_causal` (causal).")


if __name__ == "__main__":
    main()
