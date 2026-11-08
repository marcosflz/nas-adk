from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


class ToolInterrupt(Exception):
    """Pausa el run a la espera de una persona.

    Se lanza desde una tool (formulario, confirmación) o la devuelve un callback
    ``before_tool_call`` mediante ``ToolDecision.interrupt``. El Runner no la
    convierte en error de la tool: termina el run con ``status="interrupted"`` y
    deja la llamada sin resultado hasta que se reanude con ``resume=``.
    """

    def __init__(
        self, reason: str = "input_required", payload: dict[str, Any] | None = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.payload = payload or {}


@dataclass(frozen=True)
class ToolDecision:
    """Veredicto de un callback ``before_tool_call``."""

    action: Literal["allow", "deny", "interrupt"] = "allow"
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> ToolDecision:
        return cls("allow")

    @classmethod
    def deny(cls, reason: str = "denied") -> ToolDecision:
        return cls("deny", reason)

    @classmethod
    def interrupt(
        cls, reason: str = "approval", payload: dict[str, Any] | None = None
    ) -> ToolDecision:
        return cls("interrupt", reason, payload or {})


@dataclass(frozen=True)
class ResumeEntry:
    """Respuesta de una persona a una llamada que quedó pendiente.

    - ``resolved``: ``payload`` es el resultado de la tool (tools «humanas»).
    - ``approved``: se ejecuta la tool ahora, sin volver a consultar la política.
    - ``denied``: la tool no se ejecuta; el modelo recibe un error.
    """

    call_id: str
    status: Literal["resolved", "approved", "denied"] = "resolved"
    payload: Any = None
    reason: str = ""


__all__ = ["ResumeEntry", "ToolDecision", "ToolInterrupt"]
