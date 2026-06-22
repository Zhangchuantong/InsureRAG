# -*- coding: utf-8 -*-
"""Optional API-key authentication for FastAPI routes."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from config.settings import settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not bool(settings.api.get("auth_enabled", False)):
        return

    expected = str(settings.api.get("api_key") or "")
    if not expected:
        raise HTTPException(status_code=500, detail="API authentication is not configured.")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized.")
