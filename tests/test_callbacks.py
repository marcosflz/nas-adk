from __future__ import annotations

import pytest

from nas.adk.core.callbacks import BaseCallback, CallbackManager


class SpyCallback(BaseCallback):
    def __init__(self) -> None:
        self.events: list[str] = []

    async def before_model_call(self, agent_name: str, **kwargs: object) -> None:
        self.events.append(f"before_model:{agent_name}")

    async def after_model_call(self, agent_name: str, **kwargs: object) -> None:
        self.events.append(f"after_model:{agent_name}")

    async def before_tool_call(self, tool_name: str, **kwargs: object) -> None:
        self.events.append(f"before_tool:{tool_name}")

    async def after_tool_call(self, tool_name: str, **kwargs: object) -> None:
        self.events.append(f"after_tool:{tool_name}")

    async def on_event(self, **kwargs: object) -> None:
        self.events.append("on_event")

    async def on_error(self, **kwargs: object) -> None:
        self.events.append("on_error")


@pytest.mark.asyncio
async def test_callback_manager_add() -> None:
    mgr = CallbackManager()
    cb = SpyCallback()
    mgr.add(cb)
    assert len(mgr.callbacks) == 1


@pytest.mark.asyncio
async def test_callback_manager_run() -> None:
    mgr = CallbackManager()
    cb = SpyCallback()
    mgr.add(cb)

    await mgr.run("before_model_call", agent_name="test")
    await mgr.run("after_model_call", agent_name="test")
    await mgr.run("before_tool_call", tool_name="get_weather")
    await mgr.run("on_event")

    assert cb.events == [
        "before_model:test",
        "after_model:test",
        "before_tool:get_weather",
        "on_event",
    ]


@pytest.mark.asyncio
async def test_callback_manager_init_with_list() -> None:
    cb = SpyCallback()
    mgr = CallbackManager()
    mgr.add(cb)
    assert len(mgr.callbacks) == 1
