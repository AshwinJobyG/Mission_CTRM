"""Thin httpx wrapper around the JIRA REST API.

Owns base URL, auth header, per-call timeout, and maps HTTP status codes to the
typed error taxonomy. A module-level singleton is provided via ``get_client()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .auth import build_auth_header
from .config import Settings, load_settings
from .errors import (
    AuthError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
    UpstreamError,
)


@dataclass
class Response:
    json: Any
    status: int
    elapsed_ms: int


class JiraClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **build_auth_header(self.settings),
        }
        self._http = httpx.Client(
            base_url=self.settings.api_base,
            headers=headers,
            timeout=self.settings.timeout_s,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "JiraClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _raise_for_status(self, resp: httpx.Response) -> None:
        code = resp.status_code
        if code < 400:
            return
        if code in (401, 403):
            raise AuthError("Authentication/authorization failed.", status=code)
        if code == 404:
            raise NotFoundError("Resource not found.", status=code)
        if code == 429:
            raise RateLimitError("Rate limited by JIRA.", status=code)
        raise UpstreamError(f"Unexpected JIRA response ({code}).", status=code)

    def get(self, path: str, params: dict | None = None) -> Response:
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict | None = None) -> Response:
        return self._request("POST", path, json=json)

    def _request(self, method: str, path: str, **kwargs) -> Response:
        start = time.perf_counter()
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"JIRA request timed out after {self.settings.timeout_ms} ms.",
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Connection error reaching JIRA: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        self._raise_for_status(resp)
        try:
            body = resp.json()
        except ValueError:
            body = None
        return Response(json=body, status=resp.status_code, elapsed_ms=elapsed_ms)


_client: JiraClient | None = None


def get_client() -> JiraClient:
    """Return a lazily-built, process-wide client singleton."""
    global _client
    if _client is None:
        _client = JiraClient()
    return _client
