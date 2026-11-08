"""Guard de compartición: ciertos proveedores solo son aptos para uso personal y local."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def assert_shareable(
    agent: Any,
    *,
    allow_personal_use_only: bool = False,
    api_key: str | None = None,
    host: str | None = None,
) -> None:
    """Rechaza servir (A2A/MCP) agentes cuyo árbol use un proveedor ``personal_use_only``.

    ``claude-code`` usa la suscripción personal del titular: las condiciones de Anthropic no
    permiten enrutar peticiones de terceros por ella. Con ``allow_personal_use_only=True`` el
    titular asume que solo sirve a sus propios flujos y se exigen, como mínimo:

    - ``host`` en loopback (si se conoce);
    - ``api_key`` no vacía (si el protocolo la admite: pasa ``None`` cuando no hay autenticación).

    Esto limita la superficie; no puede comprobar el origen de los prompts. Que las peticiones
    sean tuyas sigue siendo responsabilidad del titular.
    """
    seen: set[int] = set()
    stack = [agent]
    providers: set[str] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        model = getattr(current, "model", None)
        get_caps = getattr(model, "get_capabilities", None)
        caps = get_caps() if callable(get_caps) else None
        if caps is not None and caps.personal_use_only:
            providers.add(str(getattr(model, "provider", type(model).__name__)))
        stack.extend(getattr(current, "sub_agents", None) or [])
    if not providers:
        return
    name = ", ".join(sorted(providers))
    if not allow_personal_use_only:
        raise ValueError(
            f"provider={name!r} is for personal local use only and cannot be exposed through "
            "A2A/MCP by default (it could route third-party requests through a personal "
            "subscription). For your own internal automations pass "
            "allow_personal_use_only=True (loopback only, API key required), or use "
            "provider='anthropic' to serve an agent."
        )
    if host is not None and host not in LOOPBACK:
        raise ValueError(
            f"provider={name!r} may only be served on loopback (127.0.0.1/localhost/::1), "
            f"not on {host!r}"
        )
    if api_key is not None and not api_key:
        raise ValueError(
            f"provider={name!r} requires an API key when served (set api_key or NAS_ADK_API_KEY)"
        )
    logger.warning(
        "Serving a %s agent: personal-use-only provider; keep it to your own automations", name
    )
