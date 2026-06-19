"""Enterprise Knowledge Retention & Discovery Assistant (PS-019).

A local RAG pipeline: read any files from a folder, embed them into a local
Chroma vector DB, and answer questions grounded in that context using a
locally hosted LLM (Ollama).
"""

__all__ = [
    "config",
    "loader",
    "chunker",
    "embeddings",
    "vectorstore",
    "ingest",
    "rag",
]
