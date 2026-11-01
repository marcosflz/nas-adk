from __future__ import annotations

import secrets
from typing import Any, TYPE_CHECKING

from starlette.requests import Request

if TYPE_CHECKING:
    from fastapi import FastAPI


def install_bearer_auth(app: FastAPI, api_key: str, public_paths: set[str] | None = None) -> None:
    """Token único para servidores de desarrollo, no autenticación multiusuario."""
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if api_key and request.url.path not in (public_paths or set()):
            token = request.headers.get("Authorization", "")
            if not secrets.compare_digest(token, f"Bearer {api_key}"):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


def require_network_auth(host: str, api_key: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("A bearer API key is required when listening outside localhost")
