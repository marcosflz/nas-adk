from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from nas.adk.core.events.event import Event
from nas.adk.sessions.in_memory_session_service import InMemorySessionService
from nas.adk.sessions.session import Session


class SQLiteSessionService(InMemorySessionService):
    """SQLite con conexión protegida y wrappers async sin recargar tras cada evento."""

    def __init__(self, db_path: str = "sessions.db") -> None:
        super().__init__()
        self._conn = sqlite3.connect(str(Path(db_path).resolve()), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, app_name TEXT NOT NULL, user_id TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, author TEXT NOT NULL DEFAULT '',
                content TEXT, role TEXT NOT NULL DEFAULT '', tool_calls TEXT NOT NULL DEFAULT '[]',
                tool_call_id TEXT NOT NULL DEFAULT '', final INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL, data TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id));
            CREATE INDEX IF NOT EXISTS events_session_time ON events(session_id, timestamp);
        """)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        if "data" not in columns:
            self._conn.execute("ALTER TABLE events ADD COLUMN data TEXT")
        self._conn.commit()

    def create_session(self, app_name: str, user_id: str, session_id: str = "") -> Session:
        session = Session(id=session_id or str(uuid4()), app_name=app_name, user_id=user_id)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions(id, app_name, user_id, state, created_at) VALUES(?, ?, ?, ?, ?)",
                (session.id, app_name, user_id, "{}", session.created_at.isoformat()),
            )
        return session

    def get_session(
        self,
        session_id: str,
        *,
        app_name: str | None = None,
        user_id: str | None = None,
        event_limit: int | None = None,
    ) -> Session | None:
        if event_limit is not None and event_limit < 0:
            raise ValueError("event_limit must not be negative")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            session = Session(
                id=row["id"],
                app_name=row["app_name"],
                user_id=row["user_id"],
                state=json.loads(row["state"]),
                created_at=row["created_at"],
            )
            self._check_owner(session, app_name, user_id)
            sql = "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp, rowid"
            params: tuple[Any, ...] = (session_id,)
            if event_limit is not None:
                sql = "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT ?"
                params = (session_id, event_limit)
            rows = self._conn.execute(sql, params).fetchall()
            if event_limit is not None:
                rows.reverse()
            for event in rows:
                if event["data"]:
                    session.events.append(Event.model_validate_json(event["data"]))
                else:
                    session.events.append(
                        Event(
                            id=event["id"],
                            author=event["author"],
                            content=event["content"],
                            role=event["role"],
                            tool_calls=json.loads(event["tool_calls"]),
                            tool_call_id=event["tool_call_id"],
                            final=bool(event["final"]),
                            timestamp=event["timestamp"],
                        )
                    )
            return session

    def get_or_create_session(self, app_name: str, user_id: str, session_id: str = "") -> Session:
        with self._lock:
            session = (
                self.get_session(session_id, app_name=app_name, user_id=user_id)
                if session_id
                else None
            )
            return (
                session
                if session is not None
                else self.create_session(app_name, user_id, session_id)
            )

    def append_event(self, session_id: str, event: Event) -> None:
        event.id = event.id or str(uuid4())
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO events(id, session_id, author, content, role, tool_calls,
                                   tool_call_id, final, timestamp, data)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.id,
                    session_id,
                    event.author,
                    event.content,
                    event.role,
                    json.dumps(event.tool_calls),
                    event.tool_call_id,
                    int(event.final),
                    event.timestamp.isoformat(),
                    event.model_dump_json(),
                ),
            )

    def save_state(self, session: Session) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT app_name, user_id, state FROM sessions WHERE id = ?", (session.id,)
            ).fetchone()
            if row is None:
                raise KeyError(session.id)
            if row["app_name"] != session.app_name or row["user_id"] != session.user_id:
                raise PermissionError("Session belongs to another application or user")
            state = json.dumps(session.state)
            if row["state"] != state:
                self._conn.execute(
                    "UPDATE sessions SET state = ? WHERE id = ?", (state, session.id)
                )

    async def get_or_create_session_async(
        self, app_name: str, user_id: str, session_id: str = ""
    ) -> Session:
        return await asyncio.to_thread(self.get_or_create_session, app_name, user_id, session_id)

    async def append_event_async(self, session_id: str, event: Event) -> None:
        await asyncio.to_thread(self.append_event, session_id, event)

    async def save_state_async(self, session: Session) -> None:
        await asyncio.to_thread(self.save_state, session)

    def delete_session(self, session_id: str, *, app_name: str, user_id: str) -> bool:
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT app_name, user_id FROM sessions WHERE id=?", (session_id,),
            ).fetchone()
            if row is None:
                return False
            if row["app_name"] != app_name or row["user_id"] != user_id:
                raise PermissionError("Session belongs to another application or user")
            if session_id in self._active:
                raise RuntimeError("Cannot delete an active session")
            self._conn.execute("DELETE FROM events WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            return True

    def delete_user_sessions(self, app_name: str, user_id: str) -> int:
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            ids = {row["id"] for row in self._conn.execute(
                "SELECT id FROM sessions WHERE app_name=? AND user_id=?", (app_name, user_id),
            )}
            if self._active.intersection(ids):
                raise RuntimeError("Cannot delete active sessions")
            self._conn.execute(
                """DELETE FROM events WHERE session_id IN
                    (SELECT id FROM sessions WHERE app_name=? AND user_id=?)""", (app_name, user_id),
            )
            self._conn.execute(
                "DELETE FROM sessions WHERE app_name=? AND user_id=?", (app_name, user_id),
            )
            return len(ids)

    async def delete_session_async(self, session_id: str, *, app_name: str, user_id: str) -> bool:
        return await asyncio.to_thread(
            self.delete_session, session_id, app_name=app_name, user_id=user_id,
        )

    async def delete_user_sessions_async(self, app_name: str, user_id: str) -> int:
        return await asyncio.to_thread(self.delete_user_sessions, app_name, user_id)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
