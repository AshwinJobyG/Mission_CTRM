"""MCP server for the JIRA source connector (PS-003).

Exposes the uniform connector contract as MCP tools so the (external) MCP
Controller — or any MCP client — can call the connector in isolation:

    jira_health()              -> up | degraded | down
    jira_fetch(id)             -> normalized record + provenance
    jira_search(query, scope)  -> ranked, provenance-tagged chunks

Auth uses a token read from the environment (API token for Cloud, PAT for
Server/DC) — never hardcoded. See JIRA_CONNECTOR.md and .env.jira.example.

Run (stdio transport):
    python jira_mcp_server.py
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from jira_connector import rag as _rag
from jira_connector import vectorstore as _vectorstore
from jira_connector.embeddings import EmbeddingError
from jira_connector.errors import JiraError
from jira_connector.fetch import fetch as _fetch
from jira_connector.health import health as _health
from jira_connector.index import build_index as _build_index
from jira_connector.search import search as _search

mcp = FastMCP("jira")


@mcp.tool()
def jira_health() -> dict:
    """Liveness of the JIRA connector: up | degraded | down (+ latency)."""
    return _health()


@mcp.tool()
def jira_fetch(id: str) -> dict:
    """Fetch a single ticket by key (e.g. CXC-1234) as a normalized record."""
    try:
        return _fetch(id)
    except JiraError as exc:
        return exc.to_dict()


@mcp.tool()
def jira_search(query: str, scope: dict | None = None) -> list[dict]:
    """Search JIRA with a natural-language query, returning ranked chunks.

    `scope` carries ACL/project constraints applied at the source, e.g.:
        {"projects": ["CXC"], "statuses": ["Open"], "max_results": 20}
    """
    try:
        return _search(query, scope or {})
    except JiraError as exc:
        return [exc.to_dict()]


@mcp.tool()
def jira_index(projects: list[str] | None = None, max_tickets: int = 200) -> str:
    """Build the JIRA vector DB (ION embeddings -> Chroma) for the given projects."""
    try:
        return str(_build_index(projects=projects or [], max_tickets=max_tickets, reset=True))
    except (JiraError, EmbeddingError) as exc:
        return f"[error] {exc}"


@mcp.tool()
def jira_ask(question: str, top_k: int = 6) -> dict:
    """Answer a question via vector RAG over the indexed JIRA knowledge.

    Returns the grounded answer, an overall confidence (0..1), and per-ticket
    source confidence. Run jira_index first to populate the vector DB.
    """
    try:
        ans = _rag.answer(question, top_k=top_k)
    except EmbeddingError as exc:
        return {"error": "embeddings", "message": str(exc)}
    return {
        "answer": ans.answer,
        "confidence": ans.confidence,
        "confidence_band": ans.confidence_band,
        "sources": [
            {"ticket": t, "confidence": round(sim, 4),
             "url": next((s.url for s in ans.sources if s.ticket == t), "")}
            for t, sim in ans.per_source
        ],
        "indexed_chunks": _vectorstore.count(),
    }


if __name__ == "__main__":
    mcp.run()
