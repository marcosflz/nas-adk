from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import json
import time

from nas.adk.memory.models import (
    Memory, MemoryPolicy, MemoryRunState, MemoryService,
)
from nas.adk.memory.sqlite_memory_service import SQLiteMemoryService
from nas.adk.sessions.session import Session


class Context:
    """Contexto de ejecución compartido entre agente, herramientas y callbacks.

    Proporciona acceso al estado de sesión y al servicio de memoria.
    """

    def __init__(
        self,
        session: Session,
        memory_service: MemoryService | None = None,
        artifact_service: Any = None,
        agent_runner: Any = None,
        *,
        memory_writable: bool = True,
        memory_policy: MemoryPolicy | None = None,
        memory_state: MemoryRunState | None = None,
    ) -> None:
        self._session = session
        self._memory = memory_service
        self.artifact_service = artifact_service
        self.agent_runner = agent_runner
        self._memory_writable = memory_writable
        self.memory_policy = memory_policy
        self.memory_state = memory_state

    @property
    def session(self) -> Session:
        return self._session

    @property
    def memory_scope(self) -> str:
        return json.dumps([self._session.app_name, self._session.user_id])

    @property
    def memory_source(self) -> str | None:
        return self.memory_state.source.content if self.memory_state else None

    def for_child(self, session: Session, *, allow_write: bool = False) -> Context:
        return Context(
            session, self._memory, self.artifact_service,
            memory_writable=self._memory_writable and allow_write,
            memory_policy=self.memory_policy, memory_state=self.memory_state,
        )

    def _write_allowed(self) -> None:
        if not self._memory_writable:
            raise PermissionError("This delegated context has read-only memory access")
        if self._memory is None:
            raise RuntimeError("No memory service configured")
        if isinstance(self._memory, SQLiteMemoryService) and self.memory_state is not None and (
            self.memory_state.snapshot is None
        ):
            raise RuntimeError("Memory snapshot unavailable; refusing an unsafe write")

    def _sqlite_memory(self) -> SQLiteMemoryService:
        if not isinstance(self._memory, SQLiteMemoryService):
            raise RuntimeError("This operation requires SQLiteMemoryService")
        return self._memory

    def _memory_metric(self, operation: str, started: float, error: bool = False) -> None:
        if self.memory_state is not None:
            metrics = self.memory_state.metrics
            key = f"{operation}_seconds"
            metrics[key] = metrics.get(key, 0.0) + time.monotonic() - started
            if error:
                key = f"{operation}_errors"
                metrics[key] = metrics.get(key, 0) + 1

    async def run_agent(self, agent: Any, input_data: Any) -> str:
        if self.agent_runner is None:
            raise RuntimeError("Agent execution requires a Runner Context")
        result: str = await self.agent_runner(agent, input_data, self)
        return result

    @property
    def state(self) -> dict[str, Any]:
        """Estado mutable de la sesión actual."""
        return self._session.state

    async def save_artifact(
        self, name: str, content: bytes, mime_type: str = "application/octet-stream"
    ) -> Any:
        if self.artifact_service is None:
            raise RuntimeError("No artifact service configured")
        return await self.artifact_service.save(self._session.id, name, content, mime_type)

    async def get_artifact(self, artifact_id: str) -> Any:
        if self.artifact_service is None:
            return None
        artifact = await self.artifact_service.get(artifact_id)
        if artifact is not None and artifact.session_id != self._session.id:
            raise PermissionError("Artifact belongs to another session")
        return artifact

    async def add_memory(self, key: str, value: str) -> None:
        """Guarda un dato en la memoria a largo plazo del usuario."""
        if isinstance(self._memory, SQLiteMemoryService):
            await self.remember_memory(key, value)
            return
        if self._memory is not None:
            self._write_allowed()
            await self._memory.add(self.memory_scope, key, value)

    async def remember_memory(
        self, key: str, value: str, *, category: Literal["profile", "preference", "fact"] | None = None,
        expires_at: datetime | None = None, expected_version: int | None = None,
    ) -> Memory:
        self._write_allowed()
        service = self._sqlite_memory()
        state = self.memory_state
        if expected_version is None and state is not None and state.snapshot is not None:
            expected_version = state.snapshot.versions.get(key, 0)
        started = time.monotonic()
        failed = True
        try:
            record = await service.remember(
                self.memory_scope, key, value, category=category, expires_at=expires_at,
                expected_version=expected_version, source_session_id=self.session.id,
                source_event_id=state.source.id if state else "",
                source_timestamp=state.source.timestamp if state else None,
            )
            if state is not None and state.snapshot is not None:
                state.snapshot.versions[key] = record.version
                # ponytail: explicit writes suppress this turn's batch; per-key suppression if needed.
                state.capture_suppressed = True
                state.retrieved = [memory for memory in state.retrieved if memory.key != key]
                state.retrieved.insert(0, record)
            failed = False
            return record
        finally:
            self._memory_metric("write", started, failed)

    async def get_memory(self, key: str) -> str | None:
        """Recupera un dato de la memoria del usuario."""
        if self._memory is not None:
            started = time.monotonic()
            failed = True
            try:
                result = await self._memory.get(self.memory_scope, key)
                failed = False
                return result
            finally:
                self._memory_metric("read", started, failed)
        return None

    async def search_memory(self, query: str) -> list[dict[str, object]]:
        """Busca en la memoria del usuario por coincidencia de texto."""
        if self._memory is not None:
            started = time.monotonic()
            failed = True
            try:
                result = await self._memory.search(self.memory_scope, query)
                failed = False
                return result
            finally:
                self._memory_metric("read", started, failed)
        return []

    async def list_memories(self, limit: int = 20, offset: int = 0) -> list[Memory]:
        started = time.monotonic()
        failed = True
        try:
            result = await self._sqlite_memory().list_memories(
                self.memory_scope, limit=limit, offset=offset,
            )
            failed = False
            return result
        finally:
            self._memory_metric("read", started, failed)

    async def forget_memory(self, key: str) -> bool:
        self._write_allowed()
        service = self._sqlite_memory()
        state = self.memory_state
        expected = state.snapshot.versions.get(key, 0) if state and state.snapshot else None
        started = time.monotonic()
        failed = True
        try:
            result = await service.delete(
                self.memory_scope, key, expected_version=expected,
                source_event_id=state.source.id if state else "",
                source_timestamp=state.source.timestamp if state else None,
            )
            if state is not None:
                state.capture_suppressed = True
                state.retrieved = [memory for memory in state.retrieved if memory.key != key]
                if state.snapshot is not None and expected is not None:
                    state.snapshot.versions[key] = expected + 1 if result else (expected or 1)
            failed = False
            return result
        finally:
            self._memory_metric("write", started, failed)
