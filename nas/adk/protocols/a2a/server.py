from __future__ import annotations

import os
import asyncio
from typing import Any, Literal
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from starlette.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from nas.adk import Agent, Runner
from nas.adk.protocols.http_security import install_bearer_auth, require_network_auth
from nas.adk.sessions.in_memory_session_service import InMemorySessionService


class _Part(BaseModel):
    text: str = Field(max_length=100000)


class _Message(BaseModel):
    role: Literal["user"] = "user"
    parts: list[_Part] = Field(min_length=1, max_length=100)


class _Send(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)
    sessionId: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)
    message: _Message


class _Rpc(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


def _build_agent_card(agent: Any, base_url: str = "") -> dict[str, Any]:
    return {
        "name": agent.name,
        "description": getattr(agent, "instruction", ""),
        "url": base_url,
        "version": "0.1",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [
            {
                "id": agent.name,
                "name": agent.name,
                "description": getattr(agent, "instruction", ""),
                "tags": ["adk"],
            }
        ],
    }


def create_a2a_app(
    agent: Agent,
    session_service: InMemorySessionService | None = None,
    base_url: str = "",
    api_key: str = "",
    max_tasks: int = 1000,
) -> Any:
    """Subset legacy tasks/send y tasks/get; no garantiza A2A moderno completo."""
    from fastapi import FastAPI

    if max_tasks <= 0:
        raise ValueError("max_tasks must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            await agent.aclose()
            await agent.aclose_models()

    app = FastAPI(title=f"A2A - {agent.name}", lifespan=lifespan)
    install_bearer_auth(app, api_key)
    runner = Runner(app_name="a2a", agent=agent, session_service=session_service)
    tasks: dict[str, dict[str, Any]] = {}

    @app.get("/.well-known/agent.json")
    @app.get("/a2a/agentCard")
    async def agent_card() -> JSONResponse:
        return JSONResponse(_build_agent_card(agent, base_url))

    @app.post("/a2a")
    async def a2a_rpc(body: _Rpc) -> JSONResponse:
        def error(code: int, message: str) -> JSONResponse:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": body.id, "error": {"code": code, "message": message}}
            )

        if body.method == "tasks/get":
            task = tasks.get(body.params.get("id", ""))
            if task is None:
                return error(-32001, "Task not found")
            return JSONResponse({"jsonrpc": "2.0", "id": body.id, "result": task})
        if body.method != "tasks/send":
            return error(-32601, "Method not found")
        try:
            params = _Send.model_validate(body.params)
        except ValidationError:
            return error(-32602, "Invalid task parameters")
        if params.id in tasks:
            return error(-32602, "Task ID already exists")
        if len(tasks) >= max_tasks:
            return error(-32000, "Development task store is full")
        task = {"id": params.id, "sessionId": params.sessionId, "state": "working"}
        tasks[params.id] = task
        final = None
        try:
            async for event in runner.run_async(
                "a2a_user", params.sessionId, " ".join(part.text for part in params.message.parts)
            ):
                if event.final:
                    final = event
            task.update(
                state="completed"
                if final is not None and final.status == "completed"
                else "failed",
                message={
                    "role": "agent",
                    "parts": [
                        {"text": final.content if final is not None else "No terminal response"}
                    ],
                },
            )
        except asyncio.CancelledError:
            task.update(state="cancelled")
            raise
        except Exception:
            task.update(
                state="failed",
                message={"role": "agent", "parts": [{"text": "Task execution failed"}]},
            )
        return JSONResponse({"jsonrpc": "2.0", "id": body.id, "result": task})

    return app


def start_a2a_server(
    agent: Agent,
    host: str = "127.0.0.1",
    port: int = 8010,
    base_url: str = "",
    api_key: str = "",
) -> None:
    import uvicorn

    api_key = api_key or os.getenv("NAS_ADK_API_KEY", "")
    require_network_auth(host, api_key)
    app = create_a2a_app(agent, base_url=base_url, api_key=api_key)
    uvicorn.run(app, host=host, port=port)
