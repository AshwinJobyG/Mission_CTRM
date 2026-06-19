"""Streamlit UI for the JIRA RAG assistant (PS-003 / PS-005 flow proof).

Flow: JIRA tickets -> ION embeddings -> Chroma vector DB -> semantic retrieval
with confidence -> ION LLM grounded answer with citations.

Run:
    streamlit run jira_app.py
"""

from __future__ import annotations

import streamlit as st

from jira_connector import rag, vectorstore
from jira_connector.embeddings import EmbeddingError
from jira_connector.errors import JiraError
from jira_connector.health import health
from jira_connector.index import build_index
from jira_connector.llm import LLMError

st.set_page_config(page_title="JIRA Knowledge Assistant", page_icon="🧭", layout="wide")

st.title("🧭 JIRA Escalation Context Assistant")
st.caption("Live JIRA → ION embeddings → Chroma vector DB → ION LLM, with provenance & confidence")


def _confidence_color(score: float) -> str:
    if score >= 0.75:
        return "#1a9850"   # green
    if score >= 0.5:
        return "#f9a825"   # amber
    return "#d73027"       # red


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Knowledge base")

    h = health()
    state = h.get("state", "?")
    badge = {"up": "🟢", "degraded": "🟡", "down": "🔴"}.get(state, "⚪")
    st.write(f"JIRA: {badge} **{state}**" + (f" ({h['latency_ms']} ms)" if h.get("latency_ms") else ""))

    try:
        chunk_count = vectorstore.count()
    except Exception:
        chunk_count = 0
    st.metric("Chunks in vector DB", chunk_count)

    st.divider()
    st.subheader("Build / refresh index")
    projects_raw = st.text_input("Project keys (comma-separated)", value="NGPOWER")
    max_tickets = st.number_input("Max tickets", min_value=10, max_value=2000, value=200, step=10)
    if st.button("📥 Build index", use_container_width=True):
        projects = [p.strip() for p in projects_raw.split(",") if p.strip()]
        with st.spinner("Fetching tickets, embedding, and indexing..."):
            try:
                report = build_index(projects=projects, max_tickets=int(max_tickets), reset=True)
                st.success(f"Indexed {report.stored} chunks from {report.tickets} tickets.")
            except (JiraError, EmbeddingError) as exc:
                st.error(str(exc))

    st.divider()
    top_k = st.slider("Sources per answer (top-k)", 1, 15, 6)
    model = st.text_input("ION model override (optional)", value="")


# ----------------------------- Chat -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("render"):
            turn["render"]()


def _render_answer(ans: rag.RagAnswer):
    color = _confidence_color(ans.confidence)
    st.markdown(
        f"<div style='margin:0.5rem 0'><b>Confidence:</b> "
        f"<span style='color:{color};font-weight:700'>{ans.confidence_pct}% "
        f"({ans.confidence_band})</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(ans.confidence)
    if ans.per_source:
        with st.expander("Sources & provenance", expanded=True):
            for ticket, sim in ans.per_source:
                url = next((s.url for s in ans.sources if s.ticket == ticket), "")
                fields = sorted({s.field for s in ans.sources if s.ticket == ticket})
                st.markdown(
                    f"**[{ticket}]** · {round(sim * 100)}% match · _{', '.join(fields)}_  \n[{url}]({url})"
                )


prompt = st.chat_input("Ask about your JIRA tickets…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                ans = rag.answer(prompt, top_k=top_k, model=model or None)
                st.markdown(ans.answer)
                _render_answer(ans)
                st.session_state.history.append(
                    {"role": "assistant", "content": ans.answer, "render": (lambda a=ans: _render_answer(a))}
                )
            except (EmbeddingError, LLMError) as exc:
                st.error(str(exc))
