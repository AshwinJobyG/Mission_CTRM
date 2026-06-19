"""Streamlit web UI for the knowledge assistant.

Run with:
    streamlit run app/web.py
"""

from __future__ import annotations

import streamlit as st

from . import config, vectorstore
from .embeddings import OllamaError
from .ingest import ingest_folder
from .rag import answer_question

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise Knowledge Retention & Discovery Assistant")
st.caption("PS-019 · Local RAG over your folder · Ollama + Chroma")

with st.sidebar:
    st.header("Knowledge base")
    st.text(config.summary())

    folder = st.text_input("Folder to ingest", value=str(config.DATA_DIR))
    reset = st.checkbox("Reset collection before ingest", value=False)
    if st.button("📥 Ingest folder", use_container_width=True):
        with st.spinner("Reading files, embedding, and indexing..."):
            try:
                report = ingest_folder(folder=folder, reset=reset)
                st.success(f"Indexed {report.stored} chunks from {report.documents} files.")
                if report.skipped:
                    st.info(f"Skipped {len(report.skipped)} unsupported/empty files.")
            except (OllamaError, FileNotFoundError) as exc:
                st.error(str(exc))

    try:
        st.metric("Chunks in vector DB", vectorstore.count())
    except Exception:
        st.metric("Chunks in vector DB", "—")

    top_k = st.slider("Sources per answer (top-k)", 1, 15, config.TOP_K)

if "history" not in st.session_state:
    st.session_state.history = []

# Replay chat history.
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for line in turn["sources"]:
                    st.markdown(line)

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                ans = answer_question(prompt, top_k=top_k)
                st.markdown(ans.answer)
                source_lines = []
                seen = set()
                for i, r in enumerate(ans.sources, 1):
                    if r.source in seen:
                        continue
                    seen.add(r.source)
                    source_lines.append(f"- **[{len(seen)}]** `{r.source}` (distance {r.distance:.3f})")
                if source_lines:
                    with st.expander("Sources"):
                        for line in source_lines:
                            st.markdown(line)
                st.session_state.history.append(
                    {"role": "assistant", "content": ans.answer, "sources": source_lines}
                )
            except OllamaError as exc:
                st.error(str(exc))
