from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from nas.adk import Agent
from nas.adk.models.base_llm import LlmResponse
from nas.adk.protocols.mcp import MCPClient, MCPServerConfig
from nas.adk.protocols.mcp.server import create_mcp_server
from tests.test_runner_contracts import ScriptedLlm


async def test_a2a_task_lookup_is_real_and_failures_are_not_completed() -> None:
    pytest.importorskip("fastapi")
    from nas.adk.protocols.a2a.server import create_a2a_app

    model = ScriptedLlm(
        responses=[
            LlmResponse(content="done"),
            LlmResponse(content="partial", finish_reason="length"),
        ]
    )
    app = create_a2a_app(Agent(name="root", model=model), api_key="offline")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        assert (await client.get("/.well-known/agent.json")).status_code == 401
        client.headers["Authorization"] = "Bearer offline"
        request = {
            "jsonrpc": "2.0",
            "id": "rpc",
            "method": "tasks/get",
            "params": {"id": "missing"},
        }
        assert (await client.post("/a2a", json=request)).json()["error"]["code"] == -32001
        request.update(
            method="tasks/send",
            params={"id": "task1", "sessionId": "session", "message": {"parts": [{"text": "hi"}]}},
        )
        task = (await client.post("/a2a", json=request)).json()["result"]
        assert task["state"] == "completed"
        assert (await client.post("/a2a", json=request)).json()["error"]["code"] == -32602
        request.update(method="tasks/get", params={"id": "task1"})
        assert (await client.post("/a2a", json=request)).json()["result"] == task
        request.update(
            method="tasks/send",
            params={"id": "task2", "sessionId": "session", "message": {"parts": [{"text": "hi"}]}},
        )
        assert (await client.post("/a2a", json=request)).json()["result"]["state"] == "failed"


async def test_mcp_public_server_preserves_full_schema_and_executes() -> None:
    from pydantic import BaseModel, Field

    class Item(BaseModel):
        value: int = Field(gt=0)

    # Evitar anotaciones locales diferidas para get_type_hints.
    def lookup(item):
        return {"answer": item.value}

    lookup.__annotations__ = {"item": Item, "return": dict}
    server = create_mcp_server(
        Agent(name="root", model=ScriptedLlm(), tools=[lookup]), host="127.0.0.1", port=8019,
    )
    tool = (await server.list_tools())[0]
    assert tool.inputSchema["$defs"]["Item"]["properties"]["value"]["exclusiveMinimum"] == 0
    assert await server.call_tool("lookup", {"item": {"value": 1}}) == {"answer": 1}
    with pytest.raises(Exception):
        await server.call_tool("lookup", {"item": {"value": -1}})
    assert server.settings.port == 8019


async def test_mcp_stdio_with_official_external_client(tmp_path: Path) -> None:
    tmp_path.joinpath("agent.py").write_text(
        "from nas.adk import Agent\n"
        "from nas.adk.models import ModelRouter\n"
        "def add(value: int) -> int:\n    return value + 1\n"
        "root_agent = Agent(name='offline', model=ModelRouter("
        "provider='opencode', base_url='https://example.invalid/go', "
        "model='deepseek-v4.1-flash'), tools=[add])\n",
        encoding="utf-8",
    )
    config = MCPServerConfig(command=sys.executable, args=["-m", "nas.adk", "mcp", str(tmp_path)])
    async with MCPClient(config) as client:
        tool = client.get_adk_tools()[0]
        assert (
            tool.to_llm_tool()["function"]["parameters"]["properties"]["value"]["type"] == "integer"
        )
        assert await tool.run_async(value=2) == "3"
