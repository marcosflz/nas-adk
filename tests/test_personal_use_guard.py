from __future__ import annotations

import pytest

from nas.adk import Agent
from nas.adk.models.claude_code import ClaudeCodeLlm
from nas.adk.models.router import ModelRouter
from nas.adk.protocols._sharing import assert_shareable
from nas.adk.protocols.mcp.server import create_mcp_server
from tests.fakes import FakeLlm


def test_claude_code_agents_cannot_be_served() -> None:
    agent = Agent(name="a", model=ModelRouter(provider="claude-code"))
    with pytest.raises(ValueError, match="personal local use only"):
        create_mcp_server(agent)
    with pytest.raises(ValueError, match="personal local use only"):
        assert_shareable(Agent(name="b", model=ClaudeCodeLlm()))


def test_subagent_with_claude_code_also_blocks() -> None:
    child = Agent(name="child", model=ClaudeCodeLlm())
    root = Agent(name="root", model=FakeLlm(), sub_agents=[child])
    with pytest.raises(ValueError, match="personal local use only"):
        assert_shareable(root)


def test_a2a_blocks_claude_code() -> None:
    pytest.importorskip("fastapi")
    from nas.adk.protocols.a2a.server import create_a2a_app

    with pytest.raises(ValueError, match="personal local use only"):
        create_a2a_app(Agent(name="a", model=ClaudeCodeLlm()))


def test_other_providers_are_shareable() -> None:
    assert_shareable(Agent(name="a", model=FakeLlm()))
    assert_shareable(Agent(name="a", model=ModelRouter(provider="opencode", model="m")))


def test_opt_in_requires_loopback_and_key() -> None:
    agent = Agent(name="a", model=ClaudeCodeLlm())
    assert_shareable(agent, allow_personal_use_only=True, api_key="k", host="127.0.0.1")
    assert_shareable(agent, allow_personal_use_only=True)  # MCP stdio: sin host ni clave
    with pytest.raises(ValueError, match="loopback"):
        assert_shareable(agent, allow_personal_use_only=True, api_key="k", host="0.0.0.0")
    with pytest.raises(ValueError, match="API key"):
        assert_shareable(agent, allow_personal_use_only=True, api_key="", host="127.0.0.1")


def test_mcp_opt_in_loopback_only() -> None:
    agent = Agent(name="a", model=ClaudeCodeLlm())
    assert create_mcp_server(agent, allow_personal_use_only=True) is not None
    with pytest.raises(ValueError, match="loopback"):
        create_mcp_server(agent, allow_personal_use_only=True, host="0.0.0.0")


async def test_a2a_opt_in_end_to_end_with_bearer() -> None:
    pytest.importorskip("fastapi")
    import httpx

    from nas.adk.protocols.a2a.server import create_a2a_app
    from tests.claude_code_fakes import INIT, FakeHandle, FakeTransport, final

    model = ClaudeCodeLlm(transport=FakeTransport([FakeHandle([INIT, final("hecho")])]))
    agent = Agent(name="interno", model=model)
    with pytest.raises(ValueError, match="API key"):
        create_a2a_app(agent, allow_personal_use_only=True)
    app = create_a2a_app(agent, api_key="secreto", allow_personal_use_only=True)
    body = {
        "jsonrpc": "2.0", "id": "1", "method": "tasks/send",
        "params": {"id": "t1", "sessionId": "s", "message": {"role": "user", "parts": [{"text": "hola"}]}},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/a2a", json=body)).status_code == 401
        ok = await c.post("/a2a", json=body, headers={"Authorization": "Bearer secreto"})
    assert ok.json()["result"]["message"]["parts"][0]["text"] == "hecho"
