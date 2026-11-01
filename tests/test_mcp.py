from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nas.adk.protocols.mcp.client import MCPClient, MCPServerConfig, MCPTool


@pytest.mark.asyncio
async def test_mcp_tool_parameters() -> None:
    tool = MCPTool(
        name="test_tool",
        description="A test tool",
        mcp_name="test_tool",
        mcp_input_schema={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    )
    schema = tool.to_llm_tool()
    params = schema["function"]["parameters"]
    assert "city" in params["properties"]
    assert params["required"] == ["city"]


@pytest.mark.asyncio
async def test_mcp_tool_run_without_client() -> None:
    tool = MCPTool(name="test", description="test")
    with pytest.raises(RuntimeError, match="not connected"):
        await tool.run_async(x=1)


@pytest.mark.asyncio
async def test_mcp_client_http_tools() -> None:
    from mcp import ClientSession
    from mcp.types import ListToolsResult, Tool
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def streams(url):
        yield object(), object()

    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    session.list_tools.side_effect = [
        ListToolsResult(tools=[], nextCursor="next"),
        ListToolsResult(tools=[Tool(name="weather", inputSchema={"type": "object"})]),
    ]
    client = MCPClient(MCPServerConfig(url="http://localhost:9999"))
    with (
        patch("nas.adk.protocols.mcp.client.sse_client", streams),
        patch(
            "nas.adk.protocols.mcp.client.ClientSession",
            return_value=session,
        ),
    ):
        async with client:
            assert [tool.name for tool in client.get_adk_tools()] == ["weather"]
            assert session.list_tools.call_args.kwargs == {"cursor": "next"}
        session.__aexit__.assert_awaited_once()


def test_mcp_server_config() -> None:
    config = MCPServerConfig(command="python", args=["-m", "server"])
    assert config.command == "python"
    assert config.args == ["-m", "server"]


def test_mcp_server_config_url() -> None:
    config = MCPServerConfig(url="http://localhost:8080/mcp")
    assert config.url == "http://localhost:8080/mcp"
