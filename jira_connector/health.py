"""health() -> up | degraded | down.

The cheapest proof that auth + reachability work. Never raises — always returns
a status object.

State mapping (see JIRA_CONNECTOR.md Phase 1):
  - 200 within degraded_ms                      -> up
  - 200 slower than degraded_ms, OR 429         -> degraded
  - auth failure / connection error / timeout   -> down
"""

from __future__ import annotations

from .client import get_client
from .config import load_settings
from .errors import AuthError, RateLimitError, TimeoutError, UpstreamError
from .schema import now_iso


def _status(state: str, latency_ms: int | None, detail: str | None) -> dict:
    return {
        "state": state,
        "latency_ms": latency_ms,
        "checked_at": now_iso(),
        "detail": detail,
    }


def health() -> dict:
    try:
        settings = load_settings()
        client = get_client()
    except Exception as exc:  # config error, etc.
        return _status("down", None, f"config/init error: {exc}")

    try:
        resp = client.get("/myself")
    except RateLimitError as exc:
        return _status("degraded", None, exc.message)
    except (AuthError, TimeoutError, UpstreamError) as exc:
        return _status("down", None, exc.message)
    except Exception as exc:  # last-resort safety net
        return _status("down", None, str(exc))

    if resp.elapsed_ms > settings.degraded_ms:
        return _status("degraded", resp.elapsed_ms, "slow response")
    return _status("up", resp.elapsed_ms, None)
