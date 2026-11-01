from __future__ import annotations

import pytest

from nas.adk import Agent, Runner
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse
from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)


class StreamMockModel(BaseLlm):
    model: str = "test"
    supports_streaming = True

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="Hello world")

    async def generate_stream_async(self, request: LlmRequest):
        for chunk in ["Hel", "lo ", "world"]:
            yield LlmResponse(content=chunk, type="delta")
        yield LlmResponse(content="Hello world")


@pytest.mark.asyncio
async def test_runner_streaming_yields_chunks() -> None:
    agent = Agent(
        name="test",
        model=StreamMockModel(),
        instruction="Say hello",
    )
    svc = InMemorySessionService()
    runner = Runner(app_name="test", agent=agent, session_service=svc)

    chunks: list[str] = []
    async for event in runner.run_async(
        user_id="u1", session_id="s1", message="Hi", stream=True
    ):
        if event.type == "stream" and event.content:
            chunks.append(event.content)

    assert chunks == ["Hel", "lo ", "world"]


@pytest.mark.asyncio
async def test_runner_streaming_accumulates() -> None:
    agent = Agent(
        name="test",
        model=StreamMockModel(),
        instruction="Say hello",
    )
    svc = InMemorySessionService()
    runner = Runner(app_name="test", agent=agent, session_service=svc)

    final_content = ""
    async for event in runner.run_async(
        user_id="u1", session_id="s1", message="Hi", stream=True
    ):
        if event.final and event.status == "completed":
            final_content += event.content

    assert final_content == "Hello world"
