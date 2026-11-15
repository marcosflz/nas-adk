"""Versioned session snapshots for hosts that persist sessions outside the ADK."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field

from nas.adk.core.events.event import Event
from nas.adk.sessions.session import Session


class SessionCheckpoint(BaseModel):
    schemaVersion: Literal[1] = 1
    sessionId: str
    appName: str
    userId: str
    events: list[Event] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_session(cls, session: Session) -> SessionCheckpoint:
        return cls(sessionId=session.id, appName=session.app_name, userId=session.user_id,
                   events=session.events, state=copy.deepcopy(session.state))

    def restore(self, session: Session) -> None:
        if (session.id, session.app_name, session.user_id) != (
                self.sessionId, self.appName, self.userId):
            raise PermissionError("Checkpoint belongs to another session, application or user")
        session.events = [event.model_copy(deep=True) for event in self.events]
        session.state = copy.deepcopy(self.state)
