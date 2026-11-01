from __future__ import annotations

import pytest

from nas.adk import Workflow, node
from nas.adk.agents.context import Context
from nas.adk.sessions.session import Session


@node
async def double(ctx: Context, n: int) -> int:
    return n * 2


@node
async def add_one(ctx: Context, n: int) -> int:
    return n + 1


@pytest.mark.asyncio
async def test_workflow_sequential() -> None:
    wf = Workflow(
        name="pipe",
        edges=[
            ("START", add_one),
            (add_one, double),
        ],
    )
    ctx = Context(Session())
    result = await wf.run(ctx, 5)
    assert result == 12  # (5 + 1) * 2


@pytest.mark.asyncio
async def test_workflow_single_node() -> None:
    wf = Workflow(
        name="single",
        edges=[("START", double)],
    )
    ctx = Context(Session())
    result = await wf.run(ctx, 3)
    assert result == 6


@pytest.mark.asyncio
async def test_workflow_node_without_name() -> None:
    @node
    async def greet(ctx: Context, name: str) -> str:
        return f"Hello, {name}!"

    wf = Workflow(
        name="greeter",
        edges=[("START", greet)],
    )
    ctx = Context(Session())
    result = await wf.run(ctx, "World")
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_workflow_conditional() -> None:
    @node
    async def classify(ctx: Context, text: str) -> str:
        if "hello" in text.lower():
            return "greeting"
        return "other"

    @node
    async def handle_greeting(ctx: Context, label: str) -> str:
        return f"Hi there! (route: {label})"

    @node
    async def handle_other(ctx: Context, label: str) -> str:
        return f"Unknown (route: {label})"

    wf = Workflow(
        name="classifier",
        edges=[
            ("START", classify),
            (classify, "greeting", handle_greeting),
            (classify, "other", handle_other),
        ],
    )
    ctx = Context(Session())
    result = await wf.run(ctx, "hello world")
    assert result == "Hi there! (route: greeting)"


@pytest.mark.asyncio
async def test_conditional_join_ignores_inactive_branch() -> None:
    @node
    async def classify(ctx, value):
        return "left"

    @node
    async def left(ctx, value):
        return "active"

    @node
    async def right(ctx, value):
        raise AssertionError("Inactive branch executed")

    @node
    async def join(ctx, value):
        return f"joined {value}"

    workflow = Workflow(
        edges=[
            ("START", classify),
            (classify, "left", left),
            (classify, "right", right),
            (left, join),
            (right, join),
        ]
    )
    assert await workflow.run(Context(Session()), "input") == "joined active"


@pytest.mark.asyncio
async def test_parallel_branches_and_named_aggregation() -> None:
    import asyncio

    started = []
    both = asyncio.Event()

    @node
    async def left(ctx, value):
        started.append("left")
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 1)
        return 1

    @node
    async def right(ctx, value):
        started.append("right")
        both.set()
        return 2

    left.parallel_safe = right.parallel_safe = True
    workflow = Workflow(edges=[("START", left), ("START", right)], max_concurrency=2)
    assert await workflow.run(Context(Session()), 0) == {"left": 1, "right": 2}


@pytest.mark.asyncio
async def test_workflow_validates_before_running() -> None:
    for edges in [
        [("START", "unknown")],
        [("START", double), (double, add_one), (add_one, double)],
        [("START", double, add_one, "bad")],
    ]:
        with pytest.raises(ValueError):
            await Workflow(edges=edges).run(Context(Session()), 0)


@pytest.mark.asyncio
async def test_workflow_uses_runner_context_and_agent_tool_loop() -> None:
    from nas.adk import Agent, Runner
    from nas.adk.models.base_llm import LlmResponse
    from tests.test_runner_contracts import ScriptedLlm

    @node
    async def remember(ctx, value):
        ctx.state["value"] = value
        return value

    model = ScriptedLlm(responses=[LlmResponse(content="answer")])
    agent = Agent(name="writer", model=model)
    workflow = Workflow(name="flow", edges=[("START", remember), (remember, agent)])
    runner = Runner(agent=workflow, max_model_calls=1)
    events = [event async for event in runner.run_async("user", "session", "real context")]
    assert events[-1].status == "completed" and events[-1].content == "answer"
    assert events[-1].metrics["model_calls"] == 1
    assert runner.session_service.get_session("session").state == {"value": "real context"}


@pytest.mark.asyncio
async def test_investigator_pipeline_is_offline_and_handles_unknown_topics() -> None:
    from pathlib import Path
    import runpy
    from nas.adk import Runner

    agent = runpy.run_path(str(Path(__file__).parents[1] / "ejemplos/investigador-terminal/agent.py"))["offline_agent"]
    runner = Runner(agent=agent)
    events = [event async for event in runner.run_async("user", "session", "cambio climatico")]
    assert events[-1].status == "completed" and "INFORME NAS" in events[-1].content
    assert "TENDENCIA" in events[-1].content and events[-1].metrics["model_calls"] == 0
    events = [event async for event in runner.run_async("user", "session", "tema desconocido")]
    assert events[-1].status == "completed" and events[-1].content.startswith("AYUDA:")
