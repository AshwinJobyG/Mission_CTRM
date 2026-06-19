"""Typed configuration. Secrets come from the environment only — never hardcoded.

Copy ``.env.jira.example`` to ``.env`` (auto-loaded if python-dotenv is present)
or export the variables directly.

Auth modes:
  cloud  -> JIRA Cloud, Basic auth with email + API token (REST API v3)
  server -> JIRA Server/Data Center, Bearer auth with a PAT (REST API v2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Optional convenience: load a .env file if python-dotenv is installed.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


class ConfigError(RuntimeError):
    """Raised when required configuration/secrets are missing or invalid."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    base_url: str
    auth_mode: str          # "cloud" | "server"
    email: str | None
    api_token: str | None   # cloud
    pat: str | None         # server/dc
    timeout_ms: int
    degraded_ms: int
    max_results: int
    verify_ssl: bool
    ca_bundle: str | None

    @property
    def api_version(self) -> str:
        """JIRA Cloud uses REST v3; Server/DC uses v2."""
        return "3" if self.auth_mode == "cloud" else "2"

    @property
    def verify(self) -> bool | str:
        """Value for httpx's ``verify``: a CA bundle path, or True/False.

        - JIRA_CA_BUNDLE set  -> verify against that CA bundle (recommended).
        - JIRA_VERIFY_SSL=false -> disable verification (self-signed/internal CA).
        - otherwise            -> verify with system CAs.
        """
        if self.ca_bundle:
            return self.ca_bundle
        return self.verify_ssl

    @property
    def api_base(self) -> str:
        return f"{self.base_url.rstrip('/')}/rest/api/{self.api_version}"

    @property
    def timeout_s(self) -> float:
        return self.timeout_ms / 1000.0

    def browse_url(self, key: str) -> str:
        return f"{self.base_url.rstrip('/')}/browse/{key}"


def load_settings() -> Settings:
    """Build and validate settings from the environment."""
    base_url = _env("JIRA_BASE_URL")
    if not base_url:
        raise ConfigError("JIRA_BASE_URL is required (e.g. https://acme.atlassian.net).")

    auth_mode = (_env("JIRA_AUTH_MODE", "cloud") or "cloud").lower()
    if auth_mode not in {"cloud", "server"}:
        raise ConfigError("JIRA_AUTH_MODE must be 'cloud' or 'server'.")

    email = _env("JIRA_EMAIL")
    api_token = _env("JIRA_API_TOKEN")
    pat = _env("JIRA_PAT")

    if auth_mode == "cloud":
        if not (email and api_token):
            raise ConfigError(
                "Cloud auth requires JIRA_EMAIL and JIRA_API_TOKEN (set them in the environment)."
            )
    else:  # server
        if not pat:
            raise ConfigError("Server/DC auth requires JIRA_PAT (set it in the environment).")

    try:
        timeout_ms = int(_env("JIRA_TIMEOUT_MS", "4000"))
        degraded_ms = int(_env("JIRA_DEGRADED_MS", "1500"))
        max_results = int(_env("JIRA_MAX_RESULTS", "20"))
    except ValueError as exc:
        raise ConfigError(f"Numeric config must be integers: {exc}") from exc

    if degraded_ms >= timeout_ms:
        raise ConfigError("JIRA_DEGRADED_MS must be less than JIRA_TIMEOUT_MS.")

    verify_ssl = (_env("JIRA_VERIFY_SSL", "true") or "true").lower() not in {"false", "0", "no"}
    ca_bundle = _env("JIRA_CA_BUNDLE")

    return Settings(
        base_url=base_url,
        auth_mode=auth_mode,
        email=email,
        api_token=api_token,
        pat=pat,
        timeout_ms=timeout_ms,
        degraded_ms=degraded_ms,
        max_results=max_results,
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
    )
