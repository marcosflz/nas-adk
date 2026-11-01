from __future__ import annotations

import threading
from uuid import uuid4

from nas.adk.core.events.event import Event
from nas.adk.sessions.session import Session


class InMemorySessionService:
    """Servicio de sesiones en memoria (no persistente).

    Almacena sesiones en un dict. Útil para desarrollo y tests.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._active: set[str] = set()

    def acquire_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._active:
                raise RuntimeError("A run is already active in this session")
            self._active.add(session_id)

    def release_session(self, session_id: str) -> None:
        with self._lock:
            self._active.discard(session_id)

    @staticmethod
    def _check_owner(session: Session, app_name: str | None, user_id: str | None) -> None:
        if (app_name is not None and session.app_name != app_name) or (
            user_id is not None and session.user_id != user_id
        ):
            raise PermissionError("Session belongs to another application or user")

    def create_session(self, app_name: str, user_id: str, session_id: str = "") -> Session:
        """Crea una nueva sesión."""
        session = Session(
            id=session_id or str(uuid4()),
            app_name=app_name,
            user_id=user_id,
        )
        with self._lock:
            if session.id in self._sessions:
                raise ValueError("Session already exists")
            self._sessions[session.id] = session
        return session

    def get_session(
        self,
        session_id: str,
        *,
        app_name: str | None = None,
        user_id: str | None = None,
    ) -> Session | None:
        """Obtiene una sesión por ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._check_owner(session, app_name, user_id)
            return session

    def get_or_create_session(self, app_name: str, user_id: str, session_id: str = "") -> Session:
        """Obtiene una sesión existente o crea una nueva."""
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                self._check_owner(session, app_name, user_id)
                return session
            return self.create_session(app_name, user_id, session_id)

    def append_event(self, session_id: str, event: Event) -> None:
        """Añade un evento al historial de la sesión."""
        with self._lock:
            session = self._sessions[session_id]
            session.events.append(event)

    def save_state(self, session: Session) -> None:
        with self._lock:
            stored = self._sessions[session.id]
            self._check_owner(stored, session.app_name, session.user_id)
            stored.state = session.state

    def delete_session(self, session_id: str, *, app_name: str, user_id: str) -> bool:
        """Owned erasure; refuses a session active on this service instance."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            self._check_owner(session, app_name, user_id)
            if session_id in self._active:
                raise RuntimeError("Cannot delete an active session")
            del self._sessions[session_id]
            return True

    def delete_user_sessions(self, app_name: str, user_id: str) -> int:
        """Administrative erasure; stop runs across other instances/processes first."""
        with self._lock:
            ids = [s.id for s in self._sessions.values()
                   if s.app_name == app_name and s.user_id == user_id]
            if self._active.intersection(ids):
                raise RuntimeError("Cannot delete active sessions")
            for session_id in ids:
                del self._sessions[session_id]
            return len(ids)
