"""fetch(id) -> normalized record.

Resolves a single ticket key (e.g. CXC-1234) into the public normalized record
schema (see schema.py / JIRA_CONNECTOR.md Phase 2), with a mandatory provenance
block. Raises NotFoundError for unknown keys; other failures surface as typed
errors from the client.
"""

from __future__ import annotations

from typing import Any

from .adf import to_text
from .client import JiraClient, get_client
from .config import load_settings
from .schema import (
    Comment,
    Link,
    Record,
    person,
    record_provenance,
)

# Only request the fields the schema needs.
_FIELDS = (
    "summary,status,assignee,reporter,priority,created,updated,"
    "labels,issuelinks,description,comment"
)


def _map_links(issuelinks: list[dict[str, Any]] | None) -> list[Link]:
    links: list[Link] = []
    for link in issuelinks or []:
        link_type = (link.get("type") or {}).get("name", "relates")
        other = link.get("outwardIssue") or link.get("inwardIssue")
        if other and other.get("key"):
            links.append(Link(type=link_type, id=other["key"]))
    return links


def _map_comments(comment_field: dict[str, Any] | None) -> list[Comment]:
    comments: list[Comment] = []
    for c in (comment_field or {}).get("comments", []) or []:
        comments.append(
            Comment(
                author=(c.get("author") or {}).get("displayName")
                or (c.get("author") or {}).get("name"),
                created=c.get("created"),
                body=to_text(c.get("body")),
            )
        )
    return comments


def normalize_issue(issue: dict[str, Any], browse_url: str) -> Record:
    """Map a raw JIRA issue payload to the normalized record."""
    key = issue.get("key", "")
    fields = issue.get("fields", {}) or {}

    return Record(
        id=key,
        summary=fields.get("summary"),
        description=to_text(fields.get("description")) or None,
        status=(fields.get("status") or {}).get("name"),
        assignee=person(fields.get("assignee")),
        reporter=person(fields.get("reporter")),
        priority=(fields.get("priority") or {}).get("name"),
        created=fields.get("created"),
        updated=fields.get("updated"),
        labels=list(fields.get("labels") or []),
        links=_map_links(fields.get("issuelinks")),
        comments=_map_comments(fields.get("comment")),
        provenance=record_provenance(browse_url),
    )


def fetch(id: str, client: JiraClient | None = None) -> Record:
    """Fetch a single ticket by key and return the normalized record."""
    client = client or get_client()
    settings = load_settings()
    resp = client.get(f"/issue/{id}", params={"fields": _FIELDS})
    return normalize_issue(resp.json, settings.browse_url(id))
