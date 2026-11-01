from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from nas.adk import Agent, Context, Runner
from nas.adk.cli import run_chat
from nas.adk.core.callbacks import BaseCallback
from nas.adk.memory import (
    MemoryCandidate, MemoryConflictError, MemoryExtractor, MemoryPolicy,
    SQLiteMemoryService, memory_tools,
)
from nas.adk.memory.models import MemoryRunState
from nas.adk.models.base_llm import LlmRequest, LlmResponse
from nas.adk.sessions import InMemorySessionService, Session, SQLiteSessionService
from nas.adk.core.events.event import Event
from tests.test_runner_contracts import ScriptedLlm, call
from tests.fakes import FakeLlm
from rich.console import Console


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[SQLiteMemoryService]:
    service = SQLiteMemoryService(str(tmp_path / "memory.db"))
    try:
        yield service
    finally:
        service.close()


def extraction(*items: dict[str, Any]) -> LlmResponse:
    return LlmResponse(content=json.dumps({"memories": items}), usage={"output_tokens": 5})


def fact(value: str = "Carlos", **kwargs: Any) -> dict[str, Any]:
    return {"key": "profile.name", "value": value, "category": "profile",
            "evidence": f"Me llamo {value}", **kwargs}


async def run(runner: Runner, text: str, session: str = "one", user: str = "user"):
    return [event async for event in runner.run_async(user, session, text)]


async def test_reopen_isolation_versions_and_provenance(tmp_path: Path) -> None:
    path = str(tmp_path / "persist.db")
    service = SQLiteMemoryService(path)
    scope = Context(Session(app_name="app", user_id="user")).memory_scope
    timestamp = datetime.now(timezone.utc)
    try:
        first = await service.remember(
            scope, "profile.name", "Carlos", category="profile", expected_version=0,
            source_session_id="session", source_event_id="event", source_timestamp=timestamp,
        )
        same = await service.remember(scope, "profile.name", "Carlos", category="profile")
        assert first.id == same.id and first.version == same.version == 1
        assert len(await service.list_memories(scope)) == 1
    finally:
        service.close()
    service = SQLiteMemoryService(path)
    try:
        saved = await service.get_record(scope, "profile.name")
        assert saved == first
        assert saved.source_event_id == "event" and saved.source_timestamp == timestamp
        for app, user in (("other", "user"), ("app", "other")):
            other = Context(Session(app_name=app, user_id=user)).memory_scope
            assert await service.get(other, "profile.name") is None
            assert await service.search(other, "Carlos") == []
            assert await service.list_memories(other) == []
        updated = await service.remember(scope, "profile.name", "Ana", expected_version=1)
        assert updated.id == first.id and updated.version == 2
        with pytest.raises(MemoryConflictError):
            await service.remember(scope, "profile.name", "stale", expected_version=1)
    finally:
        service.close()


async def test_two_connections_reject_lost_updates(memory: SQLiteMemoryService) -> None:
    initial = await memory.remember("scope", "name", "first")
    other = SQLiteMemoryService(memory.db_path)
    try:
        results = await asyncio.gather(
            memory.remember("scope", "name", "second", expected_version=initial.version),
            other.remember("scope", "name", "third", expected_version=initial.version),
            return_exceptions=True,
        )
        assert sum(isinstance(result, MemoryConflictError) for result in results) == 1
        saved = await memory.get_record("scope", "name")
        assert saved is not None and saved.version == 2 and saved.value in {"second", "third"}
    finally:
        other.close()


async def test_capture_atomicity_replay_and_forgetting(
    memory: SQLiteMemoryService, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = await memory.snapshot("scope")
    source_time = datetime.now(timezone.utc)
    candidate = MemoryCandidate(**fact())
    args = {"source_session_id": "s", "source_event_id": "e",
            "source_timestamp": source_time, "expected_revision": snapshot.revision}
    assert (await memory.apply_capture("scope", [candidate], **args))["accepted"] == 1
    assert (await memory.apply_capture("scope", [candidate], **args))["duplicate"] == 1
    await memory.mark_processing_failure("scope", "e")
    assert await memory.processing_status("scope", "e") == "completed"
    assert await memory.delete("scope", "profile.name", source_event_id="forget")
    assert await memory.get("scope", "profile.name") is None
    assert await memory.search("scope", "Carlos") == []
    assert await memory.list_memories("scope") == []
    assert (await memory.apply_capture("scope", [candidate], **args))["duplicate"] == 1
    row = memory._conn.execute("SELECT value FROM memories WHERE scope='scope'").fetchone()
    assert row["value"] is None
    snapshot = await memory.snapshot("scope")
    args.update(source_event_id="new", expected_revision=snapshot.revision)
    assert (await memory.apply_capture("scope", [candidate], **args))["rejected"] == 1
    snapshot = await memory.snapshot("scope")
    candidate.correction = True
    args.update(source_event_id="explicit-new", expected_revision=snapshot.revision)
    assert (await memory.apply_capture("scope", [candidate], **args))["accepted"] == 1

    before = await memory.snapshot("scope")
    original = memory._change

    def fail_second(scope: str, key: str, version: int, action: str, event_id: str) -> None:
        if key == "interests.second":
            raise sqlite3.OperationalError("simulated disk failure")
        original(scope, key, version, action, event_id)

    monkeypatch.setattr(memory, "_change", fail_second)
    with pytest.raises(sqlite3.OperationalError):
        await memory.apply_capture("scope", [
            MemoryCandidate(key="interests.first", value="Python", evidence="Python"),
            MemoryCandidate(key="interests.second", value="Rust", evidence="Rust"),
        ], source_session_id="s", source_event_id="rollback",
            source_timestamp=source_time, expected_revision=before.revision)
    assert await memory.get("scope", "interests.first") is None
    assert await memory.processing_status("scope", "rollback") is None
    assert await memory.snapshot("scope") == before


async def test_clear_expiry_and_old_captures(
    memory: SQLiteMemoryService, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    await memory.remember("scope", "profile.name", "Carlos")
    await memory.remember("other", "name", "Other")
    before = await memory.snapshot("scope")
    await memory.clear("scope")
    assert await memory.list_memories("scope") == []
    assert await memory.get("other", "name") == "Other"
    with pytest.raises(MemoryConflictError):
        await memory.apply_capture("scope", [MemoryCandidate(**fact())],
            source_session_id="s", source_event_id="stale", source_timestamp=now,
            expected_revision=before.revision)
    await memory.remember("scope", "facts.temporary", "temporary", expires_at=now + timedelta(hours=1))
    monkeypatch.setattr("nas.adk.memory.sqlite_memory_service.utc_now", lambda: now + timedelta(hours=2))
    assert await memory.get("scope", "facts.temporary") is None
    assert await memory.search("scope", "temporary") == []
    assert await memory.list_memories("scope") == []
    assert await memory.prune_expired("scope") == 1
    assert memory._conn.execute(
        "SELECT value FROM memories WHERE scope=? AND key=?", ("scope", "facts.temporary"),
    ).fetchone()[0] is None


async def test_search_pagination_and_no_raw_fts_syntax(memory: SQLiteMemoryService) -> None:
    for index in range(4):
        await memory.add("scope", f"interests.topic{index}", f"Python topic {index}")
    await memory.add("other", "name", "Python forbidden")
    assert len(await memory.search_records("scope", 'Python" OR scope:*', limit=2)) == 2
    first = await memory.search_records("scope", "Python", limit=2)
    second = await memory.search_records("scope", "Python", limit=2, offset=2)
    assert len({m.id for m in first + second}) == 4
    enabled = memory.fts_enabled
    memory.fts_enabled = False
    try:
        assert len(await memory.search("scope", "PYTHON")) == 4
        assert await memory.search("scope", '"***"') == []
    finally:
        memory.fts_enabled = enabled
    for operation in (
        memory.list_memories("scope", limit=0),
        memory.search_records("scope", "x", offset=-1),
        memory.search_records("scope", "x" * 513),
    ):
        with pytest.raises(ValueError):
            await operation


async def test_backup_migrations_validation_and_purge(memory: SQLiteMemoryService, tmp_path: Path) -> None:
    await memory.add("scope", "name", "Carlos")
    await memory.add("other", "name", "Ana")
    destination = str(tmp_path / "backup.db")
    await memory.backup_async(destination)
    with pytest.raises(FileExistsError):
        await memory.backup_async(destination)
    restored = SQLiteMemoryService(destination)
    try:
        assert await restored.get("scope", "name") == "Carlos"
    finally:
        restored.close()
    for key, value in (("bad key", "x"), ("name", ""), ("name", "  "), ("name", "x" * 2001)):
        with pytest.raises(ValueError):
            await memory.remember("scope", key, value)
    with pytest.raises(ValueError):
        await memory.remember("scope", "name", "x", expires_at=datetime.now())
    await memory.purge("scope")
    assert await memory.list_memories("scope") == []
    assert await memory.get("other", "name") == "Ana"
    for table in ("memories", "memory_scopes", "memory_changes", "memory_processing"):
        assert not memory._conn.execute(f"SELECT 1 FROM {table} WHERE scope='scope'").fetchone()
    memory._conn.execute("UPDATE memory_schema SET version=99")
    with pytest.raises(RuntimeError, match="schema"):
        SQLiteMemoryService(memory.db_path)


@pytest.mark.parametrize("sqlite", [False, True])
async def test_owned_session_erasure(tmp_path: Path, sqlite: bool) -> None:
    service = SQLiteSessionService(str(tmp_path / "sessions.db")) if sqlite else InMemorySessionService()
    try:
        one = service.create_session("app", "user", "one")
        service.append_event(one.id, Event(content="personal data"))
        service.create_session("app", "user", "two")
        service.create_session("other", "user", "other")
        with pytest.raises(PermissionError):
            service.delete_session("one", app_name="app", user_id="intruder")
        service.acquire_session("one")
        with pytest.raises(RuntimeError):
            service.delete_user_sessions("app", "user")
        assert service.get_session("two") is not None
        service.release_session("one")
        assert service.delete_session("one", app_name="app", user_id="user")
        assert service.delete_user_sessions("app", "user") == 1
        assert service.get_session("other") is not None
        if sqlite:
            assert not service._conn.execute("SELECT 1 FROM events").fetchone()
    finally:
        if sqlite:
            service.close()


async def test_auto_capture_reopen_and_new_session(tmp_path: Path) -> None:
    path = str(tmp_path / "agent.db")
    memory = SQLiteMemoryService(path)
    sessions = SQLiteSessionService(path)
    model = ScriptedLlm(responses=[LlmResponse(content="Hello"), extraction(fact())])
    runner = Runner(
        app_name="personal", agent=Agent(name="assistant", model=model), session_service=sessions,
        memory_service=memory, memory_policy=MemoryPolicy(capture="auto"),
        memory_extractor=MemoryExtractor(model),
    )
    try:
        events = await run(runner, "Me llamo Carlos")
        assert events[-1].status == "completed" and events[-1].metrics["model_calls"] == 2
        assert events[-1].metrics["memory"]["accepted"] == 1
        assert events[-1].metrics["total_usage"]["output_tokens"] == 5
        scope = Context(Session(app_name="personal", user_id="user")).memory_scope
        record = await memory.get_record(scope, "profile.name")
        assert record is not None and record.source_event_id == events[0].id
        saved = sessions.get_session("one")
        assert saved is not None and len(saved.events) == 2
    finally:
        sessions.close()
        memory.close()
    memory, sessions = SQLiteMemoryService(path), SQLiteSessionService(path)
    model = ScriptedLlm(responses=[LlmResponse(content="Carlos")])
    try:
        runner = Runner(app_name="personal", agent=Agent(model=model), session_service=sessions,
                        memory_service=memory, memory_policy=MemoryPolicy())
        events = await run(runner, "¿Cómo me llamo?", "new-session")
        data = next(m for m in model.requests[0].messages if m.get("name") == "nas_memory")
        assert json.loads(data["content"])[0]["value"] == "Carlos"
        assert events[-1].metrics["memory"]["retrieved"] == 1
        assert sum(m["content"] == "¿Cómo me llamo?" for m in model.requests[0].messages) == 1
    finally:
        sessions.close()
        memory.close()


async def test_capture_policy_corrections_and_failed_json(memory: SQLiteMemoryService) -> None:
    source = "Me llamo Carlos"
    model = ScriptedLlm(responses=[LlmResponse(content="ok"), extraction(
        fact(), {"key": "facts.password", "value": "secret", "evidence": source},
        {"key": "interests.rust", "value": "Rust", "evidence": "invented"},
    ), LlmResponse(content="ok"), extraction(fact("Ana", correction=True)),
        LlmResponse(content="ok"), LlmResponse(content="not JSON private-data")])
    runner = Runner(agent=Agent(model=model), memory_service=memory,
                    memory_policy=MemoryPolicy(capture="auto"), memory_extractor=MemoryExtractor(model))
    first = await run(runner, source)
    assert first[-1].metrics["memory"]["accepted"] == 1
    assert first[-1].metrics["memory"]["rejected"] == 2
    await run(runner, "Me llamo Ana", "two")
    scope = Context(Session(app_name="default", user_id="user")).memory_scope
    assert await memory.get(scope, "profile.name") == "Ana"
    third = await run(runner, "question", "three")
    assert third[-1].status == "completed"
    assert third[-1].metrics["memory"]["capture_status"] == "failed"
    assert "private-data" not in json.dumps(third[-1].metrics)
    assert await memory.processing_status(scope, third[0].id) == "failed"
    assert not MemoryPolicy().accepts(MemoryCandidate(
        key="profile.name", action="forget", evidence=source,
    ), source)


async def test_disabled_budget_and_retrieval_fallback(
    memory: SQLiteMemoryService, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = Context(Session(app_name="default", user_id="user")).memory_scope
    await memory.remember(scope, "profile.name", "Carlos", category="profile")
    model = ScriptedLlm(responses=[LlmResponse(content="ok")])
    await run(Runner(agent=Agent(model=model), memory_service=memory), "hello")
    assert all(m.get("name") != "nas_memory" for m in model.requests[0].messages)
    model = ScriptedLlm(responses=[LlmResponse(content="ok")])
    runner = Runner(agent=Agent(model=model), memory_service=memory, max_model_calls=1,
                    memory_policy=MemoryPolicy(capture="auto"), memory_extractor=MemoryExtractor(model))
    events = await run(runner, "hello", "two")
    assert events[-1].metrics["memory"]["capture_status"] == "budget_exhausted"
    assert len(model.requests) == 1

    async def unavailable(scope: str) -> Any:
        raise sqlite3.OperationalError("disk failed")

    monkeypatch.setattr(memory, "snapshot", unavailable)
    model = ScriptedLlm(responses=[LlmResponse(content="answer")])
    runner = Runner(agent=Agent(model=model), memory_service=memory, memory_policy=MemoryPolicy())
    events = await run(runner, "hello", "three")
    assert events[-1].status == "completed" and events[-1].metrics["memory"]["read_errors"] == 1
    assert all(m.get("name") != "nas_memory" for m in model.requests[0].messages)


async def test_memory_budget_never_breaks_current_turn(memory: SQLiteMemoryService) -> None:
    runner = Runner(agent=Agent(model=FakeLlm()), memory_service=memory, memory_policy=MemoryPolicy(),
                    max_context_tokens=80)
    await memory.remember("scope", "profile.name", "Carlos", category="profile")
    records = await memory.list_memories("scope")
    messages = runner._memory_messages(records) + [{"role": "user", "content": "current"}]
    assert runner._truncate_messages(messages) == [{"role": "user", "content": "current"}]


async def test_explicit_tools_and_no_reextraction(memory: SQLiteMemoryService) -> None:
    tools = memory_tools()
    for tool in tools:
        assert "ctx" not in tool.to_llm_tool()["function"]["parameters"]["properties"]
        assert "user_id" not in tool.to_llm_tool()["function"]["parameters"]["properties"]
    model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("remember", {
            "key": "profile.name", "value": "Carlos", "evidence": "Me llamo Carlos",
        })]), LlmResponse(content="saved"),
        LlmResponse(tool_calls=[call("forget_memory", {
            "key": "profile.name", "evidence": "Olvida mi nombre",
        })]), LlmResponse(content="forgotten"),
    ])
    runner = Runner(agent=Agent(model=model, tools=tools), memory_service=memory,
                    memory_policy=MemoryPolicy(capture="auto"), memory_extractor=MemoryExtractor(model))
    first = await run(runner, "Me llamo Carlos")
    assert first[-1].metrics["memory"]["capture_status"] == "suppressed_by_explicit_tool"
    second = await run(runner, "Olvida mi nombre", "two")
    assert second[-1].metrics["memory"]["capture_status"] == "suppressed_by_explicit_tool"
    assert all(m.get("name") != "nas_memory" for m in model.requests[-1].messages)
    scope = Context(Session(app_name="default", user_id="user")).memory_scope
    assert await memory.get(scope, "profile.name") is None
    assert await memory.processing_status(scope, second[0].id) == "completed"


@pytest.mark.parametrize("allow_write", [False, True])
async def test_delegated_memory_permissions(memory: SQLiteMemoryService, allow_write: bool) -> None:
    async def save(ctx: Context) -> str:
        await ctx.add_memory("profile.name", "Carlos")
        return "saved"

    child_model = ScriptedLlm(responses=[LlmResponse(tool_calls=[call("save", {})]), LlmResponse(content="done")])
    child = Agent(name="child", model=child_model, tools=[save])
    root_model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("request_task_child", {"task": "remember"})]),
        LlmResponse(content="done"),
    ])
    runner = Runner(agent=Agent(model=root_model, sub_agents=[child]), memory_service=memory,
                    memory_policy=MemoryPolicy(delegate_write=allow_write))
    events = await run(runner, "Me llamo Carlos")
    saved = runner.session_service.get_session("one")
    assert saved is not None and events[-1].status == "completed"
    result = next(event for event in saved.events if event.author == "save")
    assert ("read-only" not in (result.content or "")) == allow_write
    scope = Context(Session(app_name="default", user_id="user")).memory_scope
    assert await memory.get(scope, "profile.name") == ("Carlos" if allow_write else None)
    context = Context(Session(), memory, memory_writable=False,
                      memory_state=MemoryRunState(source=Event(content="Me llamo Carlos")))
    with pytest.raises(PermissionError):
        await context.for_child(Session(), allow_write=True).add_memory("name", "unsafe")


def test_configuration_is_explicit(memory: SQLiteMemoryService) -> None:
    with pytest.raises(ValueError, match="SQLite"):
        Runner(memory_policy=MemoryPolicy())
    with pytest.raises(ValueError, match="Extractor"):
        Runner(memory_service=memory, memory_policy=MemoryPolicy(capture="auto"))
    with pytest.raises(ValueError, match="capture"):
        Runner(memory_service=memory, memory_extractor=MemoryExtractor(ScriptedLlm()))


async def test_no_fts_build_falls_back_and_can_upgrade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = sqlite3.connect

    class WithoutFTS(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            return super().execute(sql, parameters)

    path = str(tmp_path / "without-fts.db")
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *a, **kw: original(*a, **kw, factory=WithoutFTS))
        service = SQLiteMemoryService(path)
        try:
            assert not service.fts_enabled
            await service.add("scope", "interests.python", "Python")
            await service.add("other", "name", "Python")
            assert len(await service.search("scope", "PYTHON")) == 1
            await service.delete("scope", "interests.python")
            assert await service.search("scope", "Python") == []
            await service.add("scope", "interests.rust", "Rust")
        finally:
            service.close()
    service = SQLiteMemoryService(path)
    try:
        assert len(await service.search("scope", "Rust")) == 1
        assert await service.search("scope", "Python") == []
    finally:
        service.close()


async def test_new_evidence_blocks_older_forget(memory: SQLiteMemoryService) -> None:
    now = datetime.now(timezone.utc)
    await memory.remember("scope", "profile.name", "Carlos", source_event_id="old",
                          source_timestamp=now - timedelta(seconds=5))
    reaffirmed = await memory.remember("scope", "profile.name", "Carlos", source_event_id="new",
                                      source_timestamp=now - timedelta(seconds=1))
    assert reaffirmed.version == 2 and reaffirmed.source_event_id == "new"
    snapshot = await memory.snapshot("scope")
    result = await memory.apply_capture("scope", [MemoryCandidate(
        key="profile.name", action="forget", evidence="Olvida mi nombre",
    )], source_session_id="s", source_event_id="older-forget",
        source_timestamp=now - timedelta(seconds=3), expected_revision=snapshot.revision)
    assert result["rejected"] == 1 and await memory.get("scope", "profile.name") == "Carlos"


async def test_capture_concurrent_change_preserves_answer(memory: SQLiteMemoryService) -> None:
    scope = Context(Session(app_name="default", user_id="user")).memory_scope

    class ConcurrentWrite(BaseCallback):
        async def before_model_call(self, agent_name, messages, tools):
            if agent_name == "__memory__":
                await memory.add(scope, "interests.python", "Python")

    model = ScriptedLlm(responses=[LlmResponse(content="answer"), extraction(fact())])
    runner = Runner(agent=Agent(model=model), memory_service=memory, callbacks=[ConcurrentWrite()],
                    memory_policy=MemoryPolicy(capture="auto"), memory_extractor=MemoryExtractor(model))
    events = await run(runner, "Me llamo Carlos")
    assert events[-1].content == "answer" and events[-1].status == "completed"
    assert events[-1].metrics["memory"]["capture_status"] == "conflict"
    assert await memory.get(scope, "profile.name") is None
    assert await memory.get(scope, "interests.python") == "Python"


async def test_capture_timeout_is_nonfatal(memory: SQLiteMemoryService) -> None:
    class BlockingExtractor(ScriptedLlm):
        async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
            if request.response_format:
                await asyncio.Event().wait()
            return await super().generate_content_async(request)

    model = BlockingExtractor(responses=[LlmResponse(content="answer")])
    runner = Runner(agent=Agent(model=model), memory_service=memory,
                    memory_policy=MemoryPolicy(capture="auto", capture_timeout=0.01),
                    memory_extractor=MemoryExtractor(model))
    events = await run(runner, "Me llamo Carlos")
    assert events[-1].status == "completed"
    assert events[-1].metrics["memory"]["capture_status"] == "failed"
    assert events[-1].metrics["model_calls"] == 2


@pytest.mark.parametrize("small_source", [False, True])
async def test_capture_input_bounds(memory: SQLiteMemoryService, small_source: bool) -> None:
    model = ScriptedLlm(responses=[LlmResponse(content="answer")])
    runner = Runner(agent=Agent(model=model), memory_service=memory,
                    max_context_tokens=16000 if small_source else 80,
                    memory_policy=MemoryPolicy(capture="auto", max_source_chars=10 if small_source else 8000),
                    memory_extractor=MemoryExtractor(model))
    events = await run(runner, "Me llamo Carlos")
    assert events[-1].status == "completed" and len(model.requests) == 1
    assert events[-1].metrics["memory"]["capture_status"] == (
        "source_too_large" if small_source else "context_too_large"
    )


async def test_chat_identity_reset_and_capture_warning(
    memory: SQLiteMemoryService, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    model = ScriptedLlm(responses=[LlmResponse(content="ok"), extraction(fact()),
                                 LlmResponse(content="Carlos"), LlmResponse(content="invalid JSON")])
    messages = iter(["Me llamo Carlos", "/reset", "¿Cómo me llamo?", "/salir"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(messages))
    sessions = InMemorySessionService()
    await run_chat(Agent(model=model), sessions, app_name="personal", user_id="carlos",
                   memory_service=memory, memory_policy=MemoryPolicy(capture="auto"),
                   memory_extractor=MemoryExtractor(model))
    scope = Context(Session(app_name="personal", user_id="carlos")).memory_scope
    assert await memory.get(scope, "profile.name") == "Carlos"
    assert {s.user_id for s in sessions._sessions.values()} == {"carlos"}
    assert len(sessions._sessions) == 2
    assert any(m.get("name") == "nas_memory" for m in model.requests[2].messages)
    assert "Memoria no guardada automáticamente: failed" in capsys.readouterr().out


async def test_closed_run_does_not_capture(memory: SQLiteMemoryService) -> None:
    model = ScriptedLlm(responses=[])
    runner = Runner(agent=Agent(model=model), memory_service=memory,
                    memory_policy=MemoryPolicy(capture="auto"), memory_extractor=MemoryExtractor(model))
    events = runner.run_async("user", "one", "Me llamo Carlos")
    first = await anext(events)
    await events.aclose()
    scope = Context(Session(app_name="default", user_id="user")).memory_scope
    assert first.role == "user" and await memory.get(scope, "profile.name") is None
    assert not model.requests


def test_offline_demo_from_another_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "ejemplos" / "memoria-sqlite" / "agent.py"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--demo", "--db", str(tmp_path / "demo.db")],
        cwd=tmp_path, encoding="utf-8", capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Te llamas Carlos" in result.stdout and "OK:" in result.stdout
