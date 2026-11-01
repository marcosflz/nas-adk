from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from nas.adk import Event
from nas.adk.sessions.sqlite_session_service import SQLiteSessionService


@pytest.fixture
def db(tmp_path: Path) -> Iterator[SQLiteSessionService]:
    service = SQLiteSessionService(db_path=str(tmp_path / "sessions.db"))
    try:
        yield service
    finally:
        service.close()


def test_create_and_get_session(db: SQLiteSessionService) -> None:
    session = db.create_session("my_app", "user_1")
    assert session.app_name == "my_app"
    assert session.user_id == "user_1"
    retrieved = db.get_session(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id


def test_get_or_create_existing(db: SQLiteSessionService) -> None:
    s1 = db.create_session("app", "u1", "fixed_id")
    s2 = db.get_or_create_session("app", "u1", "fixed_id")
    assert s1.id == s2.id


def test_append_and_retrieve_events(db: SQLiteSessionService) -> None:
    session = db.create_session("app", "u1")
    event = Event(author="user", content="Hello", role="user")
    db.append_event(session.id, event)
    retrieved = db.get_session(session.id)
    assert retrieved is not None
    assert len(retrieved.events) == 1
    assert retrieved.events[0].content == "Hello"


def test_session_persistence(db: SQLiteSessionService) -> None:
    session = db.create_session("app", "u1")
    db.append_event(session.id, Event(author="user", content="Hi", role="user"))
    db.append_event(session.id, Event(author="assistant", content="Hello!", role="assistant"))
    retrieved = db.get_session(session.id)
    assert retrieved is not None
    assert len(retrieved.events) == 2
    assert retrieved.events[0].role == "user"
    assert retrieved.events[1].role == "assistant"
