from __future__ import annotations

import json
from typing import Any

import jsonschema
from mcp.server import FastMCP
from mcp.types import TextContent, Tool

from nas.adk import Agent
from nas.adk.tools.base_tool import BaseTool
from nas.adk.tools.sub_agent_tool import SubAgentTool


class _ADKMCP(FastMCP):
    """Adaptación por métodos públicos para conservar schemas JSON arbitrarios."""

    def __init__(self, adk_tools: list[BaseTool], **kwargs: Any) -> None:
        self.adk_tools = {tool.name: tool for tool in adk_tools}
        if len(self.adk_tools) != len(adk_tools):
            raise ValueError("Duplicate MCP tool names")
        super().__init__(**kwargs)

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.to_llm_tool()["function"]["parameters"],
            )
            for tool in self.adk_tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.adk_tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        jsonschema.validate(arguments, tool.to_llm_tool()["function"]["parameters"])
        result = await tool.run_async(**arguments)
        if isinstance(result, dict):
            return result
        return [
            TextContent(
                type="text",
                text=result if isinstance(result, str) else json.dumps(result, default=str),
            )
        ]


def create_mcp_server(
    agent: Agent,
    name: str | None = None,
    instructions: str | None = None,
    allow_personal_use_only: bool = False,
    **settings: Any,
) -> FastMCP:
    from nas.adk.protocols._sharing import assert_shareable

    # MCP no tiene autenticación propia: con proveedores de uso personal solo se admite loopback.
    assert_shareable(
        agent,
        allow_personal_use_only=allow_personal_use_only,
        host=str(settings.get("host", "127.0.0.1")),
    )
    tools = [tool for tool in agent.resolve_tools() if not isinstance(tool, SubAgentTool)]
    return _ADKMCP(
        tools,
        name=name or agent.name,
        instructions=instructions or getattr(agent, "instruction", ""),
        **settings,
    )


def start_mcp_server(
    agent: Agent,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8011,
    sse_path: str = "/sse",
    message_path: str = "/messages/",
    allow_personal_use_only: bool = False,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "MCP development server is localhost-only; use an authenticated proxy for remote access"
        )
    mcp = create_mcp_server(
        agent,
        allow_personal_use_only=allow_personal_use_only,
        host=host,
        port=port,
        sse_path=sse_path,
        message_path=message_path,
    )
    mcp.run(transport=transport)  # type: ignore[arg-type]
