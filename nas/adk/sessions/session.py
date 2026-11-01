from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from nas.adk.core.events.event import Event


class Session(BaseModel):
    """Representa una conversación completa entre usuario y agente.

    Contiene el historial de eventos, el estado mutable (state)
    y metadatos como app_name y user_id.
    """

    id: str = ""
    app_name: str = ""
    user_id: str = ""
    state: dict[str, Any] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
