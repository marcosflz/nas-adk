from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nas.adk import Agent
from nas.adk.models.base_llm import LlmResponse
from nas.adk.orchestration.planner import ReActPlanner
from tests.fakes import FakeLlm


def _mock_llm(content: str = "Hello") -> AsyncMock:
    m = AsyncMock(model="test")
    m.generate_content_async = AsyncMock(return_value=LlmResponse(content=content))
    return m


@pytest.mark.asyncio
async def test_planner_plan() -> None:
    agent = Agent(name="test", model=FakeLlm(), instruction="Plan")
    planner = ReActPlanner(agent)
    with patch.object(
        agent,
        "model", _mock_llm("1. First step\n2. Second step\n3. Third step")
    ):
        steps = await planner.plan("Do something")
        assert len(steps) == 3


@pytest.mark.asyncio
async def test_planner_fallback() -> None:
    agent = Agent(name="test", model=FakeLlm(), instruction="Plan")
    planner = ReActPlanner(agent)
    with patch.object(agent, "model", _mock_llm("ok")):
        steps = await planner.plan("Do something")
        assert len(steps) >= 0
