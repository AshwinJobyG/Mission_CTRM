"""Connectivity smoke test — run BEFORE any feature code (Phase 0).

Verifies that auth works and JIRA is reachable by calling /myself.

    python -m jira_connector.smoke_test
"""

from __future__ import annotations

import sys

from .client import get_client
from .config import ConfigError, load_settings
from .errors import JiraError


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    print(f"Connecting to {settings.api_base} as {settings.auth_mode} ...")
    try:
        resp = get_client().get("/myself")
    except JiraError as exc:
        print(f"[{exc.kind}] {exc.message}", file=sys.stderr)
        return 1

    me = resp.json or {}
    name = me.get("displayName") or me.get("name") or "(unknown)"
    account = me.get("accountId") or me.get("key") or "(n/a)"
    print(f"OK ({resp.elapsed_ms} ms) — authenticated as {name} [{account}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
