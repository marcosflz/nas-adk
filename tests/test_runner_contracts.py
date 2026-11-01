from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field, ValidationError

from nas.adk import Agent, Context, Runner
from nas.adk.core.callbacks import BaseCallback
from nas.adk.core.events.event import Event
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ToolCall
from nas.adk.sessions import InMemorySessionService, SQLiteSessionService
from nas.adk.tools import FunctionTool


class ScriptedLlm(BaseLlm):
    model: str = "offline"
    responses: list[LlmResponse] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)
    supports_streaming: ClassVar[bool] = True

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request.model_copy(deep=True))
        return self.responses.pop(0)

    async def generate_stream_async(self, request: LlmRequest):
        response = await self.generate_content_async(request)
        text = response.content or ""
        for part in (text[:2], text[2:]):
            if part:
                yield LlmResponse(content=part, type="delta")
        yield response


def call(name: str, args: Any, id: str = "call1") -> ToolCall:
    return ToolCall(id=id, function={"name": name, "arguments": args})


async def collect(runner: Runner, message: str = "first", stream: bool = False) -> list[Event]:
    return [event async for event in runner.run_async("user", "session", message, stream=stream)]


class Spy(BaseCallback):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.errors: list[Exception] = []

    async def on_event(self, event: Event) -> None:
        self.events.append(event)

    async def on_error(self, error: Exception) -> None:
        self.errors.append(error)


@pytest.mark.parametrize("sqlite", [False, True])
async def test_canonical_history_and_persistence(tmp_path: Path, sqlite: bool) -> None:
    def lookup(value: int, ctx: Context) -> str:
        ctx.state["last"] = value
        return f"result={value * 2}"

    model = ScriptedLlm(
        responses=[
            LlmResponse(
                content="Looking up",
                tool_calls=[call("lookup", {"value": 2}), call("lookup", {"value": 3}, "call2")],
            ),
            LlmResponse(content="done"),
            LlmResponse(content="second done"),
        ]
    )
    service = (
        SQLiteSessionService(str(tmp_path / "history.db")) if sqlite else InMemorySessionService()
    )
    runner = Runner(
        app_name="test",
        agent=Agent(name="root", model=model, instruction="system", tools=[lookup]),
        session_service=service,
    )
    try:
        events = await collect(runner)
        await collect(runner, "second")
        assert model.requests[0].messages == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
        ]
        messages = model.requests[1].messages
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
        assert messages[2]["content"] == "Looking up" and len(messages[2]["tool_calls"]) == 2
        assert [m["content"] for m in messages[3:]] == ["result=4", "result=6"]
        assert [m["tool_call_id"] for m in messages[3:]] == ["call1", "call2"]
        assert model.requests[2].messages == messages + [
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "second"},
        ]
        saved = service.get_session("session", app_name="test", user_id="user")
        assert saved is not None and saved.state == {"last": 3}
        assert saved.events[0].timestamp == events[0].timestamp
        assert saved.events[-1].status == "completed"
        assert saved.created_at == service.get_session("session").created_at
    finally:
        if sqlite:
            service.close()


async def test_subagent_executes_tool_with_isolated_state_and_real_result() -> None:
    ran = []

    def lookup(value: int, ctx: Context) -> str:
        ran.append(value)
        ctx.state["child_only"] = True
        return "actual tool result"

    child_model = ScriptedLlm(
        responses=[
            LlmResponse(tool_calls=[call("lookup", {"value": 7})]),
            LlmResponse(content="child answer"),
        ]
    )
    child = Agent(name="child", model=child_model, tools=[lookup])
    parent_model = ScriptedLlm(
        responses=[
            LlmResponse(tool_calls=[call("request_task_child", {"task": "lookup"})]),
            LlmResponse(content="done"),
        ]
    )
    spy = Spy()
    runner = Runner(
        agent=Agent(name="root", model=parent_model, sub_agents=[child]), callbacks=[spy]
    )
    events = await collect(runner)
    assert ran == [7]
    assert child_model.requests[1].messages[-1]["content"] == "actual tool result"
    assert parent_model.requests[1].messages[-1]["content"] == "child answer"
    assert runner.session_service.get_session("session").state == {}
    assert any(e.author == "lookup" and e.parent_invocation_id for e in spy.events)
    assert events[-1].metrics["model_calls"] == 4
    await collect(
        runner, "second"
    )  # Sin respuestas restantes: error terminal, no tool results de hijos en prompt.
    assert all(m.get("content") != "actual tool result" for m in parent_model.requests[-1].messages)


async def test_delegation_validation_and_depth_budget() -> None:
    for args, depth in (({}, 4), ({"task": "valid"}, 0)):
        child_model = ScriptedLlm(responses=[LlmResponse(content="must not run")])
        child = Agent(name="child", model=child_model)
        model = ScriptedLlm(responses=[
            LlmResponse(tool_calls=[call("request_task_child", args)]),
            LlmResponse(content="handled"),
        ])
        events = await collect(Runner(agent=Agent(name="root", model=model, sub_agents=[child]), max_depth=depth))
        assert not child_model.requests
        if args:
            assert events[-1].status == "budget_exhausted"
        else:
            assert "error" in next(event.content for event in events if event.role == "tool")


async def test_stream_deltas_final_and_usage_are_not_persisted_twice() -> None:
    model = ScriptedLlm(
        responses=[LlmResponse(content="hello", usage={"prompt_tokens": 3, "completion_tokens": 2})]
    )
    runner = Runner(agent=Agent(name="root", model=model))
    events = await collect(runner, stream=True)
    assert "".join(e.content or "" for e in events if e.type == "stream") == "hello"
    assert events[-1].content == "hello" and events[-1].status == "completed"
    assert events[-1].metrics["total_usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert len(runner.session_service.get_session("session").events) == 2


@pytest.mark.parametrize("sqlite", [False, True])
async def test_session_ownership_and_duplicate_creation(tmp_path: Path, sqlite: bool) -> None:
    service = (
        SQLiteSessionService(str(tmp_path / "owner.db")) if sqlite else InMemorySessionService()
    )
    try:
        service.create_session("app", "owner", "fixed")
        with pytest.raises(PermissionError):
            service.get_or_create_session("app", "intruder", "fixed")
        with pytest.raises(PermissionError):
            service.get_or_create_session("other", "owner", "fixed")
        with pytest.raises(Exception):
            service.create_session("other", "intruder", "fixed")
        assert service.get_session("fixed").user_id == "owner"
    finally:
        if sqlite:
            service.close()


async def test_tools_validate_once_and_sync_functions_run_off_loop() -> None:
    def thread_id() -> int:
        return threading.get_ident()

    tool = FunctionTool(thread_id)
    assert await tool.run_async() != threading.get_ident()

    def add(value: int) -> int:
        return value + 1

    tool = FunctionTool(add)
    with pytest.raises(ValidationError):
        await tool.run_async(value="not an integer")
    with pytest.raises(ValidationError):
        await tool.run_async(value=1, unexpected=True)
    assert await tool.run_async(value="2") == 3


async def test_parallel_tools_are_opt_in_and_keep_result_order() -> None:
    started = []
    both = asyncio.Event()

    async def lookup(value: int) -> int:
        started.append(value)
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 1)
        return value

    model = ScriptedLlm(
        responses=[
            LlmResponse(
                tool_calls=[call("lookup", {"value": 1}), call("lookup", {"value": 2}, "call2")]
            ),
            LlmResponse(content="done"),
        ]
    )
    events = await collect(
        Runner(
            agent=Agent(name="root", model=model, tools=[FunctionTool(lookup, parallel_safe=True)])
        )
    )
    assert started == [1, 2]
    assert [event.content for event in events if event.role == "tool"] == ["1", "2"]


async def test_budget_exhaustion_is_terminal_and_failed_turn_not_reused() -> None:
    def lookup() -> str:
        return "data"

    response = LlmResponse(tool_calls=[call("lookup", {})])
    model = ScriptedLlm(responses=[response, LlmResponse(content="fresh")])
    spy = Spy()
    runner = Runner(
        agent=Agent(name="root", model=model, tools=[lookup]), max_model_calls=1, callbacks=[spy]
    )
    events = await collect(runner)
    assert events[-1].final and events[-1].status == "budget_exhausted"
    assert events[-1].metrics["model_calls"] == 1 and spy.errors
    assert (await collect(runner, "new"))[-1].status == "completed"
    assert model.requests[-1].messages == [{"role": "user", "content": "new"}]


async def test_deadline_cancellation_and_same_session_guard() -> None:
    entered = asyncio.Event()

    async def wait() -> str:
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    model = ScriptedLlm(
        responses=[LlmResponse(tool_calls=[call("wait", {})]), LlmResponse(content="next")]
    )
    runner = Runner(agent=Agent(name="root", model=model, tools=[wait]), timeout=1)
    task = asyncio.create_task(collect(runner))
    await asyncio.wait_for(entered.wait(), 1)
    with pytest.raises(RuntimeError, match="already active"):
        await collect(runner)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.session_service.get_session("session").events[-1].status == "cancelled"
    assert (await collect(runner))[-1].status == "completed"
    model.responses = [LlmResponse(tool_calls=[call("wait", {})])]
    runner._timeout = 0.01
    assert (await collect(runner))[-1].status == "budget_exhausted"


def test_truncation_preserves_current_whole_turn() -> None:
    runner = Runner(max_context_messages=4)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call"}]},
        {"role": "tool", "content": "result", "tool_call_id": "call"},
    ]
    assert runner._truncate_messages(messages) == [messages[0], *messages[3:]]


class Input(BaseModel):
    value: int


class Output(BaseModel):
    answer: int


async def test_agent_input_and_output_schema_enforced() -> None:
    model = ScriptedLlm(responses=[LlmResponse(content='{"answer":42}')])
    agent = Agent(name="structured", model=model, input_schema=Input, output_schema=Output)
    runner = Runner(agent=agent)
    assert (await collect(runner, '{"value":1}'))[-1].content == '{"answer":42}'
    assert model.requests[0].response_format["json_schema"]["schema"] == Output.model_json_schema()
    assert (await collect(runner, '{"value":"invalid"}'))[-1].status == "error"
    assert len(model.requests) == 1
    model.responses = [LlmResponse(content="not json")]
    assert (await collect(runner, '{"value":2}'))[-1].status == "error"


async def test_large_tool_result_keeps_full_event_but_bounds_prompt() -> None:
    def large() -> str:
        return "x" * 1000

    model = ScriptedLlm(
        responses=[LlmResponse(tool_calls=[call("large", {})]), LlmResponse(content="done")]
    )
    runner = Runner(agent=Agent(name="root", model=model, tools=[large]), max_tool_result_chars=50)
    events = await collect(runner)
    result = next(event for event in events if event.role == "tool")
    assert len(result.content) == 1000
    assert len(model.requests[-1].messages[-1]["content"]) < 200
    assert result.id in model.requests[-1].messages[-1]["content"]


async def test_context_memory_and_artifacts_are_scoped() -> None:
    from nas.adk.memory import InMemoryMemoryService
    from nas.adk.services.artifacts import InMemoryArtifactService
    from nas.adk.sessions import Session

    memory, artifacts = InMemoryMemoryService(), InMemoryArtifactService()
    first = Context(Session(id="one", app_name="app", user_id="user"), memory, artifacts)
    other = Context(Session(id="two", app_name="other", user_id="user"), memory, artifacts)
    await first.add_memory("key", "value")
    assert await first.get_memory("key") == "value" and await other.get_memory("key") is None
    artifact = await first.save_artifact("result.txt", b"full result", "text/plain")
    assert (await first.get_artifact(artifact.id)).content == b"full result"
    with pytest.raises(PermissionError):
        await other.get_artifact(artifact.id)


async def test_closing_stream_cancels_only_unfinished_runs() -> None:
    model = ScriptedLlm(responses=[LlmResponse(content="hello"), LlmResponse(content="done")])
    runner = Runner(agent=Agent(name="root", model=model))
    stream = runner.run_async("user", "session", "hi", stream=True)
    await anext(stream)
    assert (await anext(stream)).type == "stream"
    await stream.aclose()
    assert runner.session_service.get_session("session").events[-1].status == "cancelled"
    stream = runner.run_async("user", "session", "new")
    await anext(stream)
    assert (await anext(stream)).final
    await stream.aclose()
    assert runner.session_service.get_session("session").events[-1].status == "completed"
