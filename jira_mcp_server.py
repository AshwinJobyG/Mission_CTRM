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

from jira_connector.errors import JiraError
from jira_connector.fetch import fetch as _fetch
from jira_connector.health import health as _health
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


if __name__ == "__main__":
    mcp.run()
