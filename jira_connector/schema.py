"""The connector's PUBLIC CONTRACT: normalized record + chunk shapes.

Keep these stable — downstream stages (vector index, context assembler) depend
on them. See JIRA_CONNECTOR.md Phase 2/3 for the canonical documentation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict


def now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Provenance(TypedDict, total=False):
    source: str          # always "jira"
    url: str             # human-browsable ticket URL
    retrieved_at: str    # ISO-8601 UTC
    ticket: str          # chunk-level: ticket key
    field: str           # chunk-level: which field the chunk came from


class Person(TypedDict):
    name: str | None
    account_id: str | None


class Comment(TypedDict):
    author: str | None
    created: str | None
    body: str


class Link(TypedDict):
    type: str
    id: str


class Record(TypedDict):
    id: str
    summary: str | None
    description: str | None
    status: str | None
    assignee: Person | None
    reporter: Person | None
    priority: str | None
    created: str | None
    updated: str | None
    labels: list[str]
    links: list[Link]
    comments: list[Comment]
    provenance: Provenance


class Chunk(TypedDict):
    chunk_id: str
    text: str
    score: float
    provenance: Provenance


def record_provenance(url: str) -> Provenance:
    return Provenance(source="jira", url=url, retrieved_at=now_iso())


def chunk_provenance(ticket: str, field: str, url: str) -> Provenance:
    return Provenance(
        source="jira",
        ticket=ticket,
        field=field,
        url=url,
        retrieved_at=now_iso(),
    )


def empty_record(key: str, url: str) -> Record:
    """A record with all keys present (consumers never branch on missing keys)."""
    return Record(
        id=key,
        summary=None,
        description=None,
        status=None,
        assignee=None,
        reporter=None,
        priority=None,
        created=None,
        updated=None,
        labels=[],
        links=[],
        comments=[],
        provenance=record_provenance(url),
    )


def person(node: dict[str, Any] | None) -> Person | None:
    if not node:
        return None
    return Person(
        name=node.get("displayName") or node.get("name"),
        account_id=node.get("accountId") or node.get("key"),
    )
