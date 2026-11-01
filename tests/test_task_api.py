from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from nas.adk import Agent, Runner
from nas.adk.models.base_llm import LlmResponse, ToolCall
from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)
from tests.fakes import FakeLlm


class ResearchInput(BaseModel):
    topic: str
    depth: str = "standard"


class ResearchOutput(BaseModel):
    summary: str
    confidence: str


def _mock_llm(*, content: str | None = None, tool_calls: list | None = None):
    tc = tool_calls or []
    return AsyncMock(
        model="test",
        generate_content_async=AsyncMock(
            return_value=LlmResponse(
                content=content,
                tool_calls=[ToolCall(id=call.id, function=call.function) for call in tc],
            )
        ),
    )


def test_sub_agent_tool_generated() -> None:
    sub = Agent(
        name="researcher",
        model=FakeLlm(),
        instruction="Research the topic.",
    )
    parent = Agent(
        name="coordinator",
        model=FakeLlm(),
        instruction="Coordinate tasks.",
        sub_agents=[sub],
    )
    tools = parent._resolve_tools()
    tool_names = [t.name for t in tools]
    assert "request_task_researcher" in tool_names


def test_sub_agent_tool_with_input_schema() -> None:
    sub = Agent(
        name="researcher",
        model=FakeLlm(),
        instruction="Research.",
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
    )
    parent = Agent(
        name="coordinator",
        model=FakeLlm(),
        instruction="Coordinate.",
        sub_agents=[sub],
    )
    tools = parent._resolve_tools()
    tool = next(t for t in tools if t.name == "request_task_researcher")
    schema = tool.to_llm_tool()
    params = schema["function"]["parameters"]
    assert "topic" in params["properties"]
    assert params["required"] == ["topic"]


@pytest.mark.asyncio
async def test_sub_agent_runs_in_runner() -> None:
    sub = Agent(
        name="researcher",
        model=FakeLlm(),
        instruction="Research",
    )
    parent = Agent(
        name="coordinator",
        model=FakeLlm(),
        instruction="Coordinate.",
        sub_agents=[sub],
    )
    svc = InMemorySessionService()
    runner = Runner(app_name="test", agent=parent, session_service=svc)

    mock_tc = type(
        "tc",
        (object,),
        {
            "id": "call_1",
            "function": {
                "name": "request_task_researcher",
                "arguments": {"task": "research AI"},
            },
        },
    )()

    mock_parent_llm = AsyncMock(
        model="test",
        generate_content_async=AsyncMock(
            side_effect=[
                LlmResponse(tool_calls=[ToolCall(id=mock_tc.id, function=mock_tc.function)]),
                LlmResponse(content="Done"),
            ]
        ),
    )
    mock_sub_llm = AsyncMock(
        model="test",
        generate_content_async=AsyncMock(return_value=LlmResponse(content="Result")),
    )

    with (
        patch.object(parent, "model", mock_parent_llm),
        patch.object(sub, "model", mock_sub_llm),
    ):
        events = []
        async for ev in runner.run_async(user_id="u1", session_id="s1", message="Research AI"):
            events.append(ev)

        tool_events = [e for e in events if e.role == "tool"]
        assert len(tool_events) >= 1
        final = [e for e in events if e.final]
        assert len(final) > 0


@pytest.mark.asyncio
async def test_sub_agent_with_input_schema_runs() -> None:
    sub = Agent(
        name="researcher",
        model=FakeLlm(),
        instruction="Research the topic in depth.",
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
    )
    parent = Agent(
        name="coordinator",
        model=FakeLlm(),
        instruction="Use the researcher agent.",
        sub_agents=[sub],
    )
    svc = InMemorySessionService()
    runner = Runner(app_name="test", agent=parent, session_service=svc)

    with (
        patch.object(
            parent,
            "model",
            _mock_llm(
                tool_calls=[
                    type(
                        "tc",
                        (object,),
                        {
                            "id": "call_1",
                            "function": {
                                "name": "request_task_researcher",
                                "arguments": '{"topic": "AI", "depth": "deep"}',
                            },
                        },
                    )()
                ]
            ),
        ),
        patch.object(
            sub,
            "model",
            _mock_llm(content='{"summary": "AI is...", "confidence": "high"}'),
        ),
    ):
        events = []
        async for ev in runner.run_async(user_id="u1", session_id="s1", message="Research AI"):
            events.append(ev)

        assert any(e.role == "tool" for e in events)
