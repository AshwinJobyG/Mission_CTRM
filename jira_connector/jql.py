"""Natural-language query + scope -> JQL.

24h baseline: keyword matching (``text ~``) plus scope filters. ``scope`` carries
ACL/project constraints which are injected into the JQL so out-of-scope data is
never fetched (access control at the source — see JIRA_CONNECTOR.md Phase 3).

The lexical hit-set produced here is also the candidate pool a later semantic
re-ranker would reorder; keep this layer simple and swappable.
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "on", "in", "of", "for", "to", "is", "are", "was", "were",
    "and", "or", "with", "about", "latest", "what", "whats", "show", "me", "any",
    "this", "that", "from", "by", "at", "as", "be", "it", "its", "do", "does",
}

_PHRASE_RE = re.compile(r'"([^"]+)"')
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# JIRA issue keys, e.g. NGPOWER-46, CXC-1234.
_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def extract_keys(query: str) -> list[str]:
    """Pull explicit JIRA issue keys (e.g. NGPOWER-46) from the query."""
    seen: list[str] = []
    for k in _KEY_RE.findall(query):
        if k not in seen:
            seen.append(k)
    return seen


def _escape(value: str) -> str:
    """Escape a value for use inside a JQL double-quoted string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _quoted_list(values: list[str]) -> str:
    return ", ".join(f'"{_escape(v)}"' for v in values)


def extract_terms(query: str) -> list[str]:
    """Pull quoted phrases (kept whole) and meaningful keywords from the query."""
    phrases = _PHRASE_RE.findall(query)
    remainder = _PHRASE_RE.sub(" ", query)
    words = [
        w for w in _WORD_RE.findall(remainder)
        if w.lower() not in _STOPWORDS and len(w) > 1
    ]
    return phrases + words


def build_jql(query: str, scope: dict | None = None) -> str:
    """Build a JQL string from a natural-language query and a scope dict.

    scope keys (all optional):
      projects:  list[str]  -> project IN (...)
      statuses:  list[str]  -> status IN (...)
      extra_jql: str        -> appended verbatim (advanced ACL constraints)
    """
    scope = scope or {}
    clauses: list[str] = []

    # Explicit issue keys are matched by key (text ~ won't find a key in the
    # body); keep them out of the keyword text clause.
    keys = extract_keys(query)
    terms = [t for t in extract_terms(query) if t not in keys]

    match_clauses: list[str] = []
    if terms:
        text = _escape(" ".join(terms))
        match_clauses.append(f'text ~ "{text}"')
    if keys:
        match_clauses.append(f"key IN ({_quoted_list(keys)})")
    if match_clauses:
        # OR the keyword and key matches so either path can satisfy the query.
        joined = " OR ".join(match_clauses)
        clauses.append(f"({joined})" if len(match_clauses) > 1 else joined)

    projects = scope.get("projects")
    if projects:
        clauses.append(f"project IN ({_quoted_list(projects)})")

    statuses = scope.get("statuses")
    if statuses:
        clauses.append(f"status IN ({_quoted_list(statuses)})")

    extra = scope.get("extra_jql")
    if extra:
        clauses.append(f"({extra})")

    where = " AND ".join(clauses) if clauses else "ORDER BY updated DESC"
    if clauses:
        where += " ORDER BY updated DESC"
    return where
