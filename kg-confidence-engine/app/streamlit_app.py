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
from src.confidence import WEIGHTS, score_confidence  # noqa: E402
from src.corpus import Corpus  # noqa: E402
from src.decision import NODE_ID_RE, synthesize_decision  # noqa: E402
from src.graph_builder import GraphBoostedRetriever, build_context_map  # noqa: E402
from src.retrieval import build_retrievers  # noqa: E402

DATA_DIR = ROOT / "data"

# colour nodes by source tier
TIER_COLOR = {
    "resolution": "#2ca02c", "runbook": "#17becf", "doc": "#1f77b4",
    "incident": "#d62728", "ticket": "#ff7f0e", "comment": "#7f7f7f",
}


# ---- pure helpers (importable / testable without streamlit) -----------------

def context_map_html(G, cited: set[str], height: str = "520px") -> str:
    """Render the context map as an interactive pyvis HTML string."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#222222")
    net.barnes_hut(spring_length=120)
    for n in G:
        d = G.nodes[n]
        border = "#111111" if n in cited else TIER_COLOR.get(d["source_tier"], "#999")
        size = 14 + 22 * d.get("hubness", 0.0)
        title = (f"{d['title']}<br>type={d['type']} | status={d['status']} | "
                 f"tier={d['source_tier']} | {d['date']}<br>"
                 f"in-degree={d.get('in_degree',0)} hubness={d.get('hubness',0):.2f} "
                 f"fresh={d.get('freshness',0):.2f}")
        net.add_node(n, label=n, title=title, color={
            "background": TIER_COLOR.get(d["source_tier"], "#999"),
            "border": border}, borderWidth=4 if n in cited else 1, size=size)
    for u, v, data in G.edges(data=True):
        rel = data.get("rel", "")
        color = "#d62728" if rel == "contradicts" else "#aaaaaa"
        net.add_edge(u, v, label=rel, color=color, arrows="to")
    return net.generate_html(notebook=False)


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
    def _corpus():
        return Corpus.load()

    @st.cache_resource
    def _retriever(role: str):
        view = filtered_corpus(_corpus(), role)
        return view, GraphBoostedRetriever(build_retrievers(view)["hybrid"], view)

    corpus = _corpus()
    st.title("Graph-Informed Retrieval & Confidence Engine")
    st.caption("PS-019 PoC — retrieve interrelated references, build a context "
               "map, reason over its structure, and emit a confidence score "
               "derived from the subgraph (not self-reported by the LLM).")

    tab_demo, tab_rigor = st.tabs(["🔎 Demo", "📊 Results (rigor)"])

    with tab_demo:
        c1, c2 = st.columns([3, 1])
        query = c1.text_input(
            "Critical issue / question",
            "what is the root cause of the SG settlement batch failures?")
        role = c2.selectbox("Role", list(ROLE_CLEARANCES), index=2)
        n_seeds = c2.slider("retrieved (N)", 4, 12, 8)

        view, retriever = _retriever(role)
        ranked = retriever.retrieve(query, k=n_seeds)
        top = [nid for nid, _ in ranked]
        G = build_context_map(view, top, query=query)
        decision = synthesize_decision(G, query)
        conf = score_confidence(G, decision)
        cited = set(decision.cited_node_ids)

        hidden = [n["id"] for n in corpus if not can_see(n, role)]
        st.info(f"Role **{role}** can see {len(view)}/{len(corpus)} nodes. "
                f"Hidden: {len(hidden)} — these literally do not exist for this "
                f"query (enforced in code at the retrieval boundary).")

        # Stage 1 — retrieved references (the search stage)
        st.subheader("Stage 1 · Retrieved references (search)")
        st.table([{"rank": i + 1, "id": nid, "score": round(s, 4),
                   "type": view.nodes[nid]["type"], "title": view.nodes[nid]["title"]}
                  for i, (nid, s) in enumerate(ranked)])

        # Stage 2 — context map (the wow moment)
        st.subheader("Stage 2 · Context map (everything is interrelated)")
        st.caption("Node size = hub-ness (in-degree); colour = source tier; "
                   "thick black border = cited in the decision; red edges = contradictions.")
        components.html(context_map_html(G, cited), height=540, scrolling=True)

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
        st.caption("Run `python -m src.eval_confidence full` to regenerate these "
                   "artifacts.")


if __name__ == "__main__":
    main()
