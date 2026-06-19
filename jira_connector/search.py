"""search(query, scope) -> ranked chunks.

Pipeline: NL query + scope -> JQL (ACL applied at the source) -> JIRA search ->
normalize issues -> chunk long fields with provenance -> rank -> return chunks
sorted by relevance. Results are cached briefly by (jql, max_results).
"""

from __future__ import annotations

from .cache import TTLCache
from .client import JiraClient, get_client
from .config import load_settings
from .fetch import _FIELDS, normalize_issue
from .chunk import chunks_from_record
from .jql import build_jql
from .rank import score_chunks
from .schema import Chunk

_cache = TTLCache(ttl_seconds=60.0)
_FIELD_LIST = _FIELDS.split(",")


def search(query: str, scope: dict | None = None, client: JiraClient | None = None) -> list[Chunk]:
    """Return ranked, provenance-tagged chunks for a natural-language query."""
    client = client or get_client()
    settings = load_settings()
    scope = scope or {}

    max_results = int(scope.get("max_results", settings.max_results))
    jql = build_jql(query, scope)

    cache_key = f"{jql}::{max_results}::{query}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # NOTE: POST /search works on Server v2 and most Cloud instances. Newest
    # JIRA Cloud is migrating to POST /search/jql — swap the path here if needed.
    resp = client.post(
        "/search",
        json={"jql": jql, "fields": _FIELD_LIST, "maxResults": max_results},
    )

    issues = (resp.json or {}).get("issues", [])
    all_chunks: list[Chunk] = []
    updated_by_ticket: dict[str, str] = {}

    for issue in issues:
        record = normalize_issue(issue, settings.browse_url(issue.get("key", "")))
        if record.get("updated"):
            updated_by_ticket[record["id"]] = record["updated"]
        all_chunks.extend(chunks_from_record(record))

    ranked = score_chunks(all_chunks, query, updated_by_ticket)
    _cache.set(cache_key, ranked)
    return ranked
