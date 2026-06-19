"""JIRA MCP source connector (ION Hackathon PS-003).

A standalone MCP source over a live JIRA Cloud/Server instance implementing the
project's uniform connector contract:

    search(query, scope) -> ranked chunks
    fetch(id)            -> normalized record
    health()             -> up | degraded | down

See JIRA_CONNECTOR.md for the full design and the public schema contract.
"""

from .health import health
from .fetch import fetch
from .search import search

__all__ = ["health", "fetch", "search"]
