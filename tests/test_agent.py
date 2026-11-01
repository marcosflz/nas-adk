from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nas.adk import Agent, Runner
from nas.adk.models.base_llm import LlmResponse
from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)
from tests.fakes import FakeLlm


def get_weather(city: str) -> str:
    """Get the current weather."""
    return f"Sunny in {city}"


MOCK_RESPONSE_NO_TOOL = {
    "message": {"content": "The weather is nice!", "role": "assistant"},
}
MOCK_RESPONSE_WITH_TOOL = {
    "message": {
        "content": None,
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Madrid"},
                }
            }
        ],
    }
}


@pytest.mark.asyncio
async def test_agent_simple_response() -> None:
    agent = Agent(
        name="test_agent",
        model=FakeLlm(),
        instruction="You are helpful.",
    )
    assert agent.name == "test_agent"


@pytest.mark.asyncio
async def test_agent_with_tool() -> None:
    agent = Agent(
        name="weather_agent",
        model=FakeLlm(),
        instruction="Use tools.",
        tools=[get_weather],
    )
    assert len(agent.tools) == 1


@pytest.mark.asyncio
async def test_runner_simple() -> None:
    agent = Agent(
        name="test",
        model=FakeLlm(),
        instruction="You are helpful.",
    )
    svc = InMemorySessionService()
    runner = Runner(app_name="test", agent=agent, session_service=svc)

    with patch.object(
        agent,
        "model",
        new=AsyncMock(
            model="test-model",
            generate_content_async=AsyncMock(return_value=LlmResponse(content="Hello!")),
        ),
    ):
        events = []
        async for ev in runner.run_async(user_id="u1", session_id="s1", message="Hi"):
            events.append(ev)

        assert len(events) >= 2
        assert events[0].role == "user"
        assert events[-1].final
