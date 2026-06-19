"""Convert JIRA field values to plain text.

JIRA Cloud (REST v3) returns rich fields (description, comment bodies) as
Atlassian Document Format (ADF) JSON; Server/DC (v2) returns plain strings.
This helper accepts either and returns plain text.
"""

from __future__ import annotations

from typing import Any


def to_text(value: Any) -> str:
    """Flatten a string or an ADF document node into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _walk(value).strip()
    if isinstance(value, list):
        return "\n".join(to_text(v) for v in value).strip()
    return str(value)


def _walk(node: dict) -> str:
    node_type = node.get("type")
    parts: list[str] = []

    if node_type == "text":
        parts.append(node.get("text", ""))

    for child in node.get("content", []) or []:
        if isinstance(child, dict):
            parts.append(_walk(child))

    text = "".join(parts)

    # Block-level nodes get a trailing newline so paragraphs stay separated.
    if node_type in {"paragraph", "heading", "blockquote", "listItem", "codeBlock"}:
        text += "\n"
    return text
