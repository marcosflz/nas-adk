from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, cast
from uuid import uuid4

from nas.adk.memory.models import (
    Memory, MemoryCandidate, MemoryConflictError, MemorySnapshot, utc_now,
)

class SQLiteMemoryService:
    """Scoped long-term memory on local disk. Call close() when all runs finish.

    `user_id` is an opaque scope for compatibility; Context supplies the canonical
    JSON [app_name, user_id]. Direct service calls are trusted application code.
    """

    def __init__(self, db_path: str = "memory.db", *, busy_timeout: float = 5) -> None:
        if busy_timeout <= 0 or busy_timeout > 60:
            raise ValueError("busy_timeout must be in (0, 60] seconds")
        self.db_path = str(Path(db_path).resolve())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, timeout=busy_timeout, check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function(
            "casefold", 1, lambda value: value.casefold() if value is not None else "",
            deterministic=True,
        )
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.execute("PRAGMA secure_delete = ON")
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _migrate(self) -> None:
        with self._transaction():
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_schema (version INTEGER NOT NULL)"
            )
            row = self._conn.execute("SELECT version FROM memory_schema").fetchone()
            if row is not None and row["version"] != 1:
                raise RuntimeError("Unsupported memory schema version; upgrade the ADK")
            if row is None:
                for sql in (
                    """CREATE TABLE memory_scopes (
                        scope TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0)""",
                    """CREATE TABLE memories (
                        id TEXT NOT NULL UNIQUE, scope TEXT NOT NULL, key TEXT NOT NULL,
                        value TEXT, category TEXT NOT NULL, version INTEGER NOT NULL,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, expires_at TEXT,
                        source_session_id TEXT NOT NULL, source_event_id TEXT NOT NULL,
                        source_timestamp TEXT, PRIMARY KEY(scope, key))""",
                    """CREATE INDEX memories_scope_category
                        ON memories(scope, category, updated_at)""",
                    """CREATE TABLE memory_changes (
                        id INTEGER PRIMARY KEY, scope TEXT NOT NULL, key TEXT NOT NULL,
                        version INTEGER NOT NULL, action TEXT NOT NULL,
                        source_event_id TEXT NOT NULL, timestamp TEXT NOT NULL)""",
                    "CREATE INDEX memory_changes_scope ON memory_changes(scope)",
                    """CREATE TABLE memory_processing (
                        scope TEXT NOT NULL, event_id TEXT NOT NULL, status TEXT NOT NULL,
                        timestamp TEXT NOT NULL, PRIMARY KEY(scope, event_id))""",
                ):
                    self._conn.execute(sql)
                self._conn.execute("INSERT INTO memory_schema VALUES(1)")
            existed = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'memories_fts'"
            ).fetchone()
            try:
                self._conn.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        key, value, tokenize='unicode61 remove_diacritics 2')"""
                )
                self.fts_enabled = True
            except sqlite3.OperationalError as error:
                if "no such module: fts5" not in str(error).lower():
                    raise
                self.fts_enabled = False
            if not self.fts_enabled and existed:
                raise RuntimeError("This database contains FTS5 indexes; use an FTS5-enabled SQLite")
            if self.fts_enabled and not existed:
                self._conn.execute(
                    """INSERT INTO memories_fts(rowid, key, value)
                        SELECT rowid, key, value FROM memories WHERE value IS NOT NULL"""
                )

    @staticmethod
    def _check_scope(scope: str) -> None:
        if not scope or len(scope) > 1024:
            raise ValueError("Memory scope must contain 1..1024 characters")

    @staticmethod
    def _page(limit: int, offset: int) -> None:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must not be negative")

    def _revision(self, scope: str) -> int:
        row = self._conn.execute(
            "SELECT revision FROM memory_scopes WHERE scope = ?", (scope,),
        ).fetchone()
        return row["revision"] if row else 0

    def _bump(self, scope: str) -> None:
        self._conn.execute(
            """INSERT INTO memory_scopes(scope, revision) VALUES(?, 1)
                ON CONFLICT(scope) DO UPDATE SET revision = revision + 1""", (scope,),
        )

    def _snapshot(self, scope: str) -> MemorySnapshot:
        self._check_scope(scope)
        with self._lock:
            # A read transaction also protects this snapshot from OTHER connections.
            self._conn.execute("BEGIN")
            try:
                revision = self._revision(scope)
                # ponytail: O(n) key/version snapshot; use per-key snapshots for huge profiles.
                versions = {
                    row["key"]: row["version"] for row in self._conn.execute(
                        "SELECT key, version FROM memories WHERE scope = ?", (scope,),
                    )
                }
                return MemorySnapshot(revision=revision, versions=versions)
            finally:
                self._conn.rollback()

    async def snapshot(self, user_id: str) -> MemorySnapshot:
        return await asyncio.to_thread(self._snapshot, user_id)

    def _row(self, scope: str, key: str) -> sqlite3.Row | None:
        return cast(sqlite3.Row | None, self._conn.execute(
            "SELECT rowid, * FROM memories WHERE scope = ? AND key = ?", (scope, key),
        ).fetchone())

    @staticmethod
    def _record(row: sqlite3.Row) -> Memory:
        return Memory.model_validate({
            name: row[name] for name in Memory.model_fields
        })

    @staticmethod
    def _active(row: sqlite3.Row | None) -> bool:
        return row is not None and row["value"] is not None and (
            row["expires_at"] is None or row["expires_at"] > utc_now().isoformat()
        )

    def _change(self, scope: str, key: str, version: int, action: str, event_id: str) -> None:
        self._conn.execute(
            """INSERT INTO memory_changes(scope, key, version, action, source_event_id, timestamp)
                VALUES(?, ?, ?, ?, ?, ?)""",
            (scope, key, version, action, event_id, utc_now().isoformat()),
        )

    def _index(self, rowid: int, key: str, value: str | None) -> None:
        if self.fts_enabled:
            self._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (rowid,))
            if value is not None:
                self._conn.execute(
                    "INSERT INTO memories_fts(rowid, key, value) VALUES(?, ?, ?)",
                    (rowid, key, value),
                )

    def _remember(
        self, scope: str, key: str, value: str, category: Literal["profile", "preference", "fact"],
        expires_at: datetime | None, source_session_id: str, source_event_id: str,
        source_timestamp: datetime | None, expected_version: int | None,
    ) -> Memory:
        self._check_scope(scope)
        row = self._row(scope, key)
        version = row["version"] if row else 0
        if expected_version is not None and version != expected_version:
            raise MemoryConflictError("Memory version changed; reread before updating")
        now = utc_now()
        record = Memory(
            id=row["id"] if row else str(uuid4()), key=key, value=value, category=category,
            version=version + 1, created_at=row["created_at"] if row else now, updated_at=now,
            expires_at=expires_at, source_session_id=source_session_id,
            source_event_id=source_event_id, source_timestamp=source_timestamp,
        )
        if record.expires_at is not None and record.expires_at <= now:
            raise ValueError("expires_at must be in the future")
        if source_timestamp is not None and row is not None and row["source_timestamp"]:
            if record.source_timestamp is not None and (
                record.source_timestamp < datetime.fromisoformat(row["source_timestamp"])
            ):
                raise MemoryConflictError("An older event cannot replace a newer memory")
        if self._active(row) and row is not None and all((
            row["value"] == value, row["category"] == category,
            row["expires_at"] == (record.expires_at.isoformat() if record.expires_at else None),
            not source_event_id or row["source_event_id"] == source_event_id,
            record.source_timestamp is None or (
                row["source_timestamp"] == record.source_timestamp.isoformat()
            ),
        )):
            return self._record(row)
        data = record.model_dump(mode="json")
        for name in ("created_at", "updated_at", "expires_at", "source_timestamp"):
            date = getattr(record, name)
            data[name] = date.isoformat() if date is not None else None
        self._conn.execute(
            """INSERT INTO memories(id, scope, key, value, category, version, created_at,
                updated_at, expires_at, source_session_id, source_event_id, source_timestamp)
                VALUES(:id, :scope, :key, :value, :category, :version, :created_at,
                       :updated_at, :expires_at, :source_session_id, :source_event_id,
                       :source_timestamp)
                ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value,
                category=excluded.category, version=excluded.version,
                updated_at=excluded.updated_at, expires_at=excluded.expires_at,
                source_session_id=excluded.source_session_id,
                source_event_id=excluded.source_event_id, source_timestamp=excluded.source_timestamp""",
            {**data, "scope": scope},
        )
        saved = self._row(scope, key)
        assert saved is not None
        self._index(saved["rowid"], key, value)
        self._change(scope, key, record.version, "remember", source_event_id)
        self._bump(scope)
        return record

    async def remember(
        self, user_id: str, key: str, value: str, *,
        category: Literal["profile", "preference", "fact"] | None = None,
        expires_at: datetime | None = None, source_session_id: str = "",
        source_event_id: str = "", source_timestamp: datetime | None = None,
        expected_version: int | None = None,
    ) -> Memory:
        if category is None:
            category = "profile" if key.startswith("profile.") else (
                "preference" if key.startswith("preferences.") else "fact"
            )
        def write() -> Memory:
            with self._transaction():
                assert category is not None
                record = self._remember(
                    user_id, key, value, category, expires_at, source_session_id,
                    source_event_id, source_timestamp, expected_version,
                )
                if source_event_id:
                    self._mark(user_id, source_event_id, "completed")
                return record
        return await asyncio.to_thread(write)

    async def add(self, user_id: str, key: str, value: str) -> None:
        await self.remember(user_id, key, value)

    async def get_record(self, user_id: str, key: str) -> Memory | None:
        self._check_scope(user_id)
        def read() -> Memory | None:
            with self._lock:
                row = self._row(user_id, key)
                return self._record(row) if row is not None and self._active(row) else None
        return await asyncio.to_thread(read)

    async def get(self, user_id: str, key: str) -> str | None:
        record = await self.get_record(user_id, key)
        return record.value if record else None

    async def list_memories(
        self, user_id: str, *, category: Literal["profile", "preference", "fact"] | None = None,
        limit: int = 20, offset: int = 0,
    ) -> list[Memory]:
        self._check_scope(user_id)
        self._page(limit, offset)
        def read() -> list[Memory]:
            sql = """SELECT * FROM memories WHERE scope = ? AND value IS NOT NULL
                AND (expires_at IS NULL OR expires_at > ?)"""
            params: list[Any] = [user_id, utc_now().isoformat()]
            if category is not None:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY updated_at DESC, key LIMIT ? OFFSET ?"
            with self._lock:
                return [self._record(row) for row in self._conn.execute(
                    sql, (*params, limit, offset),
                )]
        return await asyncio.to_thread(read)

    async def search_records(
        self, user_id: str, query: str, *, limit: int = 20, offset: int = 0,
    ) -> list[Memory]:
        self._check_scope(user_id)
        self._page(limit, offset)
        if len(query) > 512:
            raise ValueError("Memory search query exceeds 512 characters")
        terms = re.findall(r"[^\W_]+", query, re.UNICODE)[:16]
        if not terms:
            return []
        def read() -> list[Memory]:
            params: list[Any] = [user_id, utc_now().isoformat()]
            sql = """SELECT m.* FROM memories m """
            if self.fts_enabled:
                sql += " JOIN memories_fts ON memories_fts.rowid = m.rowid "
            sql += """ WHERE m.scope = ? AND m.value IS NOT NULL
                AND (m.expires_at IS NULL OR m.expires_at > ?) AND """
            if self.fts_enabled:
                sql += " memories_fts MATCH ? ORDER BY bm25(memories_fts), m.updated_at DESC "
                params.append(" OR ".join('"' + term + '"' for term in terms))
            else:
                sql += "(" + " OR ".join(
                    "(instr(casefold(m.key), ?) > 0 OR instr(casefold(m.value), ?) > 0)"
                    for _ in terms
                ) + ") ORDER BY m.updated_at DESC "
                for term in terms:
                    params.extend([term.casefold(), term.casefold()])
            with self._lock:
                return [self._record(row) for row in self._conn.execute(
                    sql + " LIMIT ? OFFSET ?", (*params, limit, offset),
                )]
        return await asyncio.to_thread(read)

    async def search(self, user_id: str, query: str) -> list[dict[str, object]]:
        return [record.search_result() for record in await self.search_records(user_id, query)]

    def _delete(
        self, scope: str, key: str, source_event_id: str,
        source_timestamp: datetime | None, expected_version: int | None = None,
    ) -> bool:
        row = self._row(scope, key)
        version = row["version"] if row else 0
        if expected_version is not None and version != expected_version:
            raise MemoryConflictError("Memory version changed; reread before deleting")
        if row is None:
            # A tombstone also blocks an in-flight writer that expected a missing key.
            now = utc_now()
            record = Memory(
                id=str(uuid4()), key=key, value="[deleted]", version=1,
                created_at=now, updated_at=now, source_timestamp=source_timestamp,
            )
            self._conn.execute(
                """INSERT INTO memories(id, scope, key, value, category, version,
                    created_at, updated_at, expires_at, source_session_id, source_event_id,
                    source_timestamp) VALUES(?, ?, ?, NULL, 'fact', 1, ?, ?, NULL, '', ?, ?)""",
                (record.id, scope, key, now.isoformat(), now.isoformat(), source_event_id,
                 record.source_timestamp.isoformat() if record.source_timestamp else None),
            )
            self._change(scope, key, 1, "forget", source_event_id)
            self._bump(scope)
            return False
        elif row["value"] is None:
            return False
        self._index(row["rowid"], key, None)
        self._conn.execute(
            """UPDATE memories SET value=NULL, version=version+1, updated_at=?, expires_at=NULL,
                source_session_id='', source_event_id=?, source_timestamp=?
                WHERE scope=? AND key=?""",
            (utc_now().isoformat(), source_event_id,
             source_timestamp.isoformat() if source_timestamp else None, scope, key),
        )
        self._change(scope, key, row["version"] + 1, "forget", source_event_id)
        self._bump(scope)
        return version > 0 and row["value"] is not None

    async def delete(
        self, user_id: str, key: str, *, expected_version: int | None = None,
        source_event_id: str = "", source_timestamp: datetime | None = None,
    ) -> bool:
        self._check_scope(user_id)
        if source_timestamp is not None and source_timestamp.utcoffset() is None:
            raise ValueError("Source timestamp must include a timezone")
        def write() -> bool:
            with self._transaction():
                deleted = self._delete(
                    user_id, key, source_event_id, source_timestamp, expected_version,
                )
                if source_event_id:
                    self._mark(user_id, source_event_id, "completed")
                return deleted
        return await asyncio.to_thread(write)

    def _mark(self, scope: str, event_id: str, status: str) -> None:
        self._conn.execute(
            """INSERT INTO memory_processing VALUES(?, ?, ?, ?)
                ON CONFLICT(scope, event_id) DO UPDATE SET status=excluded.status,
                timestamp=excluded.timestamp WHERE memory_processing.status != 'completed'""",
            (scope, event_id, status, utc_now().isoformat()),
        )

    async def processing_status(self, user_id: str, event_id: str) -> str | None:
        self._check_scope(user_id)
        def read() -> str | None:
            with self._lock:
                row = self._conn.execute(
                    "SELECT status FROM memory_processing WHERE scope=? AND event_id=?",
                    (user_id, event_id),
                ).fetchone()
                return row["status"] if row else None
        return await asyncio.to_thread(read)

    async def mark_processing_failure(self, user_id: str, event_id: str) -> None:
        self._check_scope(user_id)
        def write() -> None:
            with self._transaction():
                self._mark(user_id, event_id, "failed")
        await asyncio.to_thread(write)

    async def apply_capture(
        self, user_id: str, candidates: list[MemoryCandidate], *, source_session_id: str,
        source_event_id: str, source_timestamp: datetime, expected_revision: int,
    ) -> dict[str, int]:
        """Atomically apply a validated capture and its processing marker.

        Policy/evidence validation belongs to the Runner; this is a trusted API.
        A scope revision rejects stale batches, including after a concurrent clear.
        """
        self._check_scope(user_id)
        if not source_event_id or source_timestamp.utcoffset() is None:
            raise ValueError("Capture requires an event ID and an aware source timestamp")
        if len(candidates) > 10 or len({c.key for c in candidates}) != len(candidates):
            raise ValueError("Capture requires at most 10 distinct keys")
        def write() -> dict[str, int]:
            with self._transaction():
                processed = self._conn.execute(
                    "SELECT status FROM memory_processing WHERE scope=? AND event_id=?",
                    (user_id, source_event_id),
                ).fetchone()
                if processed and processed["status"] == "completed":
                    return {"accepted": 0, "rejected": 0, "duplicate": 1}
                if self._revision(user_id) != expected_revision:
                    # ponytail: scope-wide CAS; use per-key capture CAS if conversations contend.
                    raise MemoryConflictError("Scope changed during capture; batch not applied")
                accepted = rejected = 0
                for candidate in candidates:
                    row = self._row(user_id, candidate.key)
                    if row is not None and row["source_timestamp"] and (
                        source_timestamp < datetime.fromisoformat(row["source_timestamp"])
                    ):
                        rejected += 1
                        continue
                    if candidate.action == "forget":
                        self._delete(user_id, candidate.key, source_event_id, source_timestamp)
                    else:
                        if row is not None and (
                            row["value"] is None or (
                                self._active(row) and row["value"] != candidate.value
                            )
                        ) and not candidate.correction:
                            rejected += 1
                            continue
                        assert candidate.value is not None
                        self._remember(
                            user_id, candidate.key, candidate.value, candidate.category, None,
                            source_session_id, source_event_id, source_timestamp, None,
                        )
                    accepted += 1
                self._mark(user_id, source_event_id, "completed")
                return {"accepted": accepted, "rejected": rejected, "duplicate": 0}
        return await asyncio.to_thread(write)

    async def clear(self, user_id: str) -> None:
        """Forget values; retain content-free tombstones and replay markers."""
        self._check_scope(user_id)
        def write() -> None:
            with self._transaction():
                rows = self._conn.execute(
                    "SELECT key FROM memories WHERE scope=? AND value IS NOT NULL", (user_id,),
                ).fetchall()
                for row in rows:
                    self._delete(user_id, row["key"], "", utc_now())
                self._bump(user_id)  # Invalidates in-flight captures even for an empty scope.
        await asyncio.to_thread(write)

    async def prune_expired(self, user_id: str) -> int:
        self._check_scope(user_id)
        def write() -> int:
            with self._transaction():
                rows = self._conn.execute(
                    """SELECT key FROM memories WHERE scope=? AND value IS NOT NULL
                        AND expires_at <= ?""", (user_id, utc_now().isoformat()),
                ).fetchall()
                for row in rows:
                    self._delete(user_id, row["key"], "", utc_now())
                return len(rows)
        return await asyncio.to_thread(write)

    async def purge(self, user_id: str) -> None:
        """Administrative erasure. Stop ALL runs for this scope first, across processes.

        Unlike clear(), removes replay markers too. Does not erase session history,
        backups, or guarantee forensic deletion from filesystem/WAL pages.
        """
        self._check_scope(user_id)
        def write() -> None:
            with self._transaction():
                for row in self._conn.execute(
                    "SELECT rowid, key FROM memories WHERE scope=?", (user_id,),
                ).fetchall():
                    self._index(row["rowid"], row["key"], None)
                for table in ("memories", "memory_changes", "memory_processing", "memory_scopes"):
                    self._conn.execute(f"DELETE FROM {table} WHERE scope=?", (user_id,))
        await asyncio.to_thread(write)

    def backup(self, destination: str) -> None:
        """Consistent backup of this database (including sessions if co-located)."""
        path = Path(destination).resolve()
        if str(path) == self.db_path or path.exists():
            raise FileExistsError("Backup destination must be a new file")
        # Exclusive creation avoids overwriting a file created after the exists check.
        with path.open("xb"):
            pass
        with self._lock, closing(sqlite3.connect(str(path))) as target:
            self._conn.backup(target)

    async def backup_async(self, destination: str) -> None:
        await asyncio.to_thread(self.backup, destination)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
