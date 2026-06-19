"""search(query, scope) -> ranked chunks.

Pipeline: NL query + scope -> JQL (ACL applied at the source) -> JIRA search ->
normalize issues -> chunk long fields with provenance -> rank -> return chunks
sorted by relevance. Results are cached briefly by (jql, max_results).
"""

from __future__ import annotations

from .cache import TTLCache
from .client import JiraClient, get_client
from .config import load_settings
from .errors import UpstreamError
from .fetch import _FIELDS, normalize_issue
from .chunk import chunks_from_record
from .jql import build_jql, extract_keys
from .rank import score_chunks
from .schema import Chunk

_cache = TTLCache(ttl_seconds=60.0)
_FIELD_LIST = _FIELDS.split(",")


def _run_search(client: JiraClient, body: dict, api_version: str):
    """POST a JQL search, using the right endpoint for the JIRA flavor.

    JIRA Cloud removed the legacy POST /rest/api/3/search (now 410 Gone) in
    favour of the enhanced POST /rest/api/3/search/jql. Server/DC (v2) still
    uses /search. We pick by version and fall back if the instance differs.
    """
    primary = "/search/jql" if api_version == "3" else "/search"
    secondary = "/search" if primary == "/search/jql" else "/search/jql"
    try:
        return client.post(primary, json=body)
    except UpstreamError as exc:
        if exc.status in (404, 410):
            return client.post(secondary, json=body)
        raise


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

    resp = _run_search(
        client,
        {"jql": jql, "fields": _FIELD_LIST, "maxResults": max_results},
        settings.api_version,
    )

    issues = (resp.json or {}).get("issues", [])
    all_chunks: list[Chunk] = []
    updated_by_ticket: dict[str, str] = {}

    for issue in issues:
        record = normalize_issue(issue, settings.browse_url(issue.get("key", "")))
        if record.get("updated"):
            updated_by_ticket[record["id"]] = record["updated"]
        all_chunks.extend(chunks_from_record(record))

    ranked = score_chunks(
        all_chunks,
        query,
        updated_by_ticket,
        boost_tickets=set(extract_keys(query)),
    )
    _cache.set(cache_key, ranked)
    return ranked
