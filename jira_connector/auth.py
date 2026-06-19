"""Auth header construction — the single seam an auth broker would replace.

In production the controller's auth broker would mint short-lived, per-user
tokens and inject them here. For the hackathon we build the header from a single
service credential read from the environment (see config.py).
"""

from __future__ import annotations

import base64

from .config import Settings


def build_auth_header(settings: Settings) -> dict[str, str]:
    """Return the Authorization header for the configured auth mode.

    Cloud  -> Basic base64(email:api_token)
    Server -> Bearer <PAT>
    """
    if settings.auth_mode == "cloud":
        raw = f"{settings.email}:{settings.api_token}".encode("utf-8")
        token = base64.b64encode(raw).decode("ascii")
        return {"Authorization": f"Basic {token}"}
    return {"Authorization": f"Bearer {settings.pat}"}
