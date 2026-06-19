"""Build the JIRA vector DB: fetch tickets in scope -> chunk -> embed -> store.

This is the ingestion half of the RAG flow from the architecture: enterprise
source (JIRA) -> chunked, provenance-tagged knowledge -> vector index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from . import vectorstore
from .chunk import chunks_from_record
from .client import JiraClient, get_client
from .config import load_settings
from .fetch import _FIELDS, normalize_issue
from .jql import build_jql
from .search import _run_search

_FIELD_LIST = _FIELDS.split(",")


@dataclass
class IndexReport:
    projects: list[str]
    tickets: int
    chunks: int
    stored: int
    total_in_db: int

    def __str__(self) -> str:
        return (
            f"Projects    : {', '.join(self.projects) or '(all)'}\n"
            f"Tickets     : {self.tickets}\n"
            f"Chunks      : {self.chunks}\n"
            f"Stored      : {self.stored}\n"
            f"Total in DB : {self.total_in_db}"
        )


def _iter_issues(
    client: JiraClient, jql: str, api_version: str, max_tickets: int
) -> Iterator[dict]:
    """Page through search results (nextPageToken for Cloud v3, startAt for v2)."""
    fetched = 0
    if api_version == "3":
        token = None
        while fetched < max_tickets:
            body = {"jql": jql, "fields": _FIELD_LIST, "maxResults": min(100, max_tickets - fetched)}
            if token:
                body["nextPageToken"] = token
            data = _run_search(client, body, api_version).json or {}
            issues = data.get("issues", [])
            for issue in issues:
                yield issue
                fetched += 1
            token = data.get("nextPageToken")
            if not token or not issues:
                break
    else:
        start = 0
        while fetched < max_tickets:
            body = {
                "jql": jql,
                "fields": _FIELD_LIST,
                "maxResults": min(100, max_tickets - fetched),
                "startAt": start,
            }
            data = _run_search(client, body, api_version).json or {}
            issues = data.get("issues", [])
            for issue in issues:
                yield issue
                fetched += 1
            start += len(issues)
            if not issues or start >= int(data.get("total", 0)):
                break


def build_index(
    projects: list[str] | None = None,
    max_tickets: int = 200,
    reset: bool = True,
    client: JiraClient | None = None,
) -> IndexReport:
    """Fetch tickets for the given projects and (re)build the vector DB."""
    client = client or get_client()
    settings = load_settings()
    projects = projects or []

    if reset:
        vectorstore.reset()

    scope = {"projects": projects} if projects else {}
    jql = build_jql("", scope)

    ticket_count = 0
    all_chunks = []
    for issue in _iter_issues(client, jql, settings.api_version, max_tickets):
        record = normalize_issue(issue, settings.browse_url(issue.get("key", "")))
        all_chunks.extend(chunks_from_record(record))
        ticket_count += 1

    stored = vectorstore.add_chunks(all_chunks)
    return IndexReport(
        projects=projects,
        tickets=ticket_count,
        chunks=len(all_chunks),
        stored=stored,
        total_in_db=vectorstore.count(),
    )
