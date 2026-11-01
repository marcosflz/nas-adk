from __future__ import annotations

from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)


def test_create_and_get_session() -> None:
    svc = InMemorySessionService()
    session = svc.create_session("my_app", "user_1")
    assert session.app_name == "my_app"
    assert session.user_id == "user_1"
    retrieved = svc.get_session(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id


def test_get_or_create_existing() -> None:
    svc = InMemorySessionService()
    s1 = svc.create_session("app", "u1", "fixed_id")
    s2 = svc.get_or_create_session("app", "u1", "fixed_id")
    assert s1.id == s2.id


def test_get_or_create_new() -> None:
    svc = InMemorySessionService()
    session = svc.get_or_create_session("app", "u1")
    assert session.id is not None
