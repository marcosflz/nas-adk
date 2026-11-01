from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Unidad de comunicación en una conversación.

    Cada interacción entre usuario, agente y herramientas genera eventos
    que forman el historial de la sesión.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    author: str = ""
    content: str | None = None
    role: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str = ""
    final: bool = False
    type: str = "message"
    status: str | None = None
    invocation_id: str = ""
    parent_invocation_id: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    response_items: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_message(self) -> dict[str, Any]:
        """Convierte el evento a formato mensaje para la API del LLM."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg
