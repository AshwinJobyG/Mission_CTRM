"""Typed error taxonomy.

The connector never leaks raw stack traces to its consumer (the controller).
Every failure is one of these typed errors with a stable ``kind`` string and a
``to_dict()`` for serialization.
"""

from __future__ import annotations


class JiraError(Exception):
    """Base class for all connector errors."""

    kind: str = "error"

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status

    def to_dict(self) -> dict:
        return {"error": self.kind, "message": self.message, "status": self.status}


class AuthError(JiraError):
    """401 / 403 — invalid or insufficient credentials."""

    kind = "auth"


class RateLimitError(JiraError):
    """429 — too many requests."""

    kind = "rate-limit"


class NotFoundError(JiraError):
    """404 / unknown key."""

    kind = "not-found"


class TimeoutError(JiraError):  # noqa: A001 - intentional domain name
    """Request exceeded the per-call timeout."""

    kind = "timeout"


class UpstreamError(JiraError):
    """5xx or any other unexpected upstream failure."""

    kind = "upstream"
