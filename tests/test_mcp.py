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
    async def streams(url, **kwargs):
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


@pytest.mark.asyncio
async def test_mcp_config_headers_filter_and_prefix() -> None:
    from contextlib import asynccontextmanager

    from mcp import ClientSession
    from mcp.types import ListToolsResult, Tool

    captured: dict[str, object] = {}

    @asynccontextmanager
    async def streams(url, **kwargs):
        captured.update(kwargs)
        yield object(), object()

    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(name="search", inputSchema={"type": "object"}),
            Tool(name="delete_repo", inputSchema={"type": "object"}),
        ]
    )
    client = MCPClient(
        MCPServerConfig(
            url="http://localhost:9999",
            transport="sse",
            headers={"Authorization": "Bearer abc"},
            name_prefix="git hub",
            allow_tools=["search"],
        )
    )
    with (
        patch("nas.adk.protocols.mcp.client.sse_client", streams),
        patch("nas.adk.protocols.mcp.client.ClientSession", return_value=session),
    ):
        async with client:
            tools = client.get_adk_tools()

    assert captured["headers"] == {"Authorization": "Bearer abc"}
    # Solo la tool permitida, con prefijo saneado; el nombre MCP original se conserva.
    assert [(t.name, t.mcp_name) for t in tools] == [("git_hub__search", "search")]


@pytest.mark.asyncio
async def test_mcp_streamable_http_sends_headers_through_an_httpx_client() -> None:
    from contextlib import asynccontextmanager

    from mcp import ClientSession
    from mcp.types import ListToolsResult

    captured: dict[str, object] = {}

    @asynccontextmanager
    async def streams(url, *, http_client=None, **kwargs):
        captured["client"] = http_client
        yield object(), object(), lambda: None

    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    session.list_tools.return_value = ListToolsResult(tools=[])
    client = MCPClient(
        MCPServerConfig(
            url="http://localhost:9999/mcp",
            transport="streamable-http",
            headers={"X-Token": "t"},
        )
    )
    with (
        patch("nas.adk.protocols.mcp.client.streamable_http_client", streams),
        patch("nas.adk.protocols.mcp.client.ClientSession", return_value=session),
    ):
        async with client:
            http = captured["client"]
            assert http is not None and http.headers["x-token"] == "t"
