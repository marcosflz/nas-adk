from __future__ import annotations

from typing import Any
from contextlib import AsyncExitStack
import copy
import re

import httpx

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from nas.adk.tools.base_tool import BaseTool


class MCPServerConfig:
    """Configuración para conectar a un servidor MCP.

    Args:
        command: Comando para lanzar el servidor vía stdio.
        args: Argumentos del comando.
        url: URL del servidor HTTP (SSE o Streamable HTTP).
        transport: "stdio" | "sse" | "streamable-http" (default auto).
        headers: Cabeceras HTTP (p. ej. ``Authorization``) para SSE y Streamable HTTP.
        auth: ``httpx.Auth`` para SSE y Streamable HTTP.
        env: Variables de entorno del proceso en transporte stdio.
        timeout: Segundos de espera de las peticiones HTTP.
        name_prefix: Prefijo de los nombres de tool (``{prefix}__{tool}``). Evita
            colisiones al combinar varios servidores en un mismo agente.
        allow_tools: Si se indica, solo se exponen estas tools (nombres originales).
    """

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        transport: str | None = None,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        name_prefix: str = "",
        allow_tools: list[str] | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.url = url
        self.transport = transport
        self.headers = dict(headers or {})
        self.auth = auth
        self.env = dict(env) if env is not None else None
        self.timeout = timeout
        self.name_prefix = name_prefix
        self.allow_tools = list(allow_tools) if allow_tools is not None else None


class MCPTool(BaseTool):
    """Tool que envuelve una herramienta expuesta por un servidor MCP."""

    model_config = {"arbitrary_types_allowed": True}  # ClientSession no es Pydantic

    session: ClientSession | None = None
    mcp_name: str = ""
    mcp_input_schema: dict[str, Any] = {}

    def _parameters_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self.mcp_input_schema)

    async def run_async(self, **kwargs: Any) -> Any:
        if self.session is None:
            raise RuntimeError("MCP session not connected")
        result = await self.session.call_tool(self.mcp_name, kwargs)
        if result.isError or result.structuredContent is not None:
            return result.model_dump(mode="json", exclude_none=True)
        texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
        return "\n".join(texts) if texts else result.model_dump()


class MCPClient:
    """Cliente MCP basado en la librería oficial.

    Conecta a servidores MCP vía stdio, SSE o Streamable HTTP,
    descubre herramientas y las expone como ADK tools.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._tools: list[dict[str, Any]] = []
        self._stack = AsyncExitStack()

    @property
    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def connect(self) -> None:
        if self._session is not None:
            raise RuntimeError("MCP client is already connected")
        transport = self._resolve_transport()
        try:
            if transport == "stdio":
                await self._connect_stdio()
            elif transport == "sse":
                await self._connect_sse()
            elif transport == "streamable-http":
                await self._connect_streamable_http()
            else:
                raise RuntimeError(f"Unsupported transport: {transport}")
        except BaseException:
            await self.close()
            raise

    def _resolve_transport(self) -> str:
        if self._config.transport:
            return self._config.transport
        if self._config.url:
            return "sse"
        if self._config.command:
            return "stdio"
        raise RuntimeError("No transport configured (set command, url, or transport)")

    async def _connect_stdio(self) -> None:
        assert self._config.command is not None
        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env,
        )
        streams = stdio_client(params)
        read, write = await self._stack.enter_async_context(streams)
        session = await self._stack.enter_async_context(ClientSession(read, write))
        self._session = session
        await session.initialize()
        await self._load_tools()

    async def _connect_sse(self) -> None:
        assert self._config.url is not None
        streams = sse_client(
            self._config.url,
            headers=self._config.headers or None,
            timeout=self._config.timeout,
            auth=self._config.auth,
        )
        read, write = await self._stack.enter_async_context(streams)
        session = await self._stack.enter_async_context(ClientSession(read, write))
        self._session = session
        await session.initialize()
        await self._load_tools()

    async def _connect_streamable_http(self) -> None:
        assert self._config.url is not None
        http_client: httpx.AsyncClient | None = None
        if self._config.headers or self._config.auth is not None:
            http_client = await self._stack.enter_async_context(
                httpx.AsyncClient(
                    headers=self._config.headers,
                    auth=self._config.auth,
                    timeout=self._config.timeout,
                    follow_redirects=True,
                )
            )
        streams = streamable_http_client(self._config.url, http_client=http_client)
        read, write, _ = await self._stack.enter_async_context(streams)
        session = await self._stack.enter_async_context(ClientSession(read, write))
        self._session = session
        await session.initialize()
        await self._load_tools()

    async def _load_tools(self) -> None:
        assert self._session is not None
        self._tools = []
        cursor = None
        seen: set[str] = set()
        while True:
            result = await self._session.list_tools(cursor=cursor)
            for t in result.tools:
                allowed = self._config.allow_tools
                if allowed is not None and t.name not in allowed:
                    continue
                self._tools.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema or {},
                    }
                )
            cursor = result.nextCursor
            if cursor is None:
                break
            if cursor in seen:
                raise RuntimeError("MCP discovery returned a repeated pagination cursor")
            seen.add(cursor)

    def get_adk_tools(self) -> list[MCPTool]:
        if self._session is None:
            return []
        prefix = self._config.name_prefix
        return [
            MCPTool(
                name=_tool_name(prefix, t["name"]),
                description=t["description"],
                mcp_name=t["name"],
                mcp_input_schema=t["input_schema"],
                session=self._session,
            )
            for t in self._tools
        ]

    async def close(self) -> None:
        try:
            await self._stack.aclose()
        finally:
            self._session = None
            self._tools.clear()

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _tool_name(prefix: str, name: str) -> str:
    """Nombre expuesto al modelo: ``{prefix}__{name}`` limitado a [A-Za-z0-9_-]{1,64}."""
    if not prefix:
        return name
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{prefix}__{name}")
    return safe[:64]
