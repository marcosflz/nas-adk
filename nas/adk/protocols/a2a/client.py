from __future__ import annotations

import uuid
from typing import Any

import httpx

from nas.adk.tools.base_tool import BaseTool


class A2AAgentCard:
    """Tarjeta de agente A2A con metadatos del agente remoto."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.name: str = data.get("name", "unknown")
        self.description: str = data.get("description", "")
        self.url: str = data.get("url", "")
        self.version: str = data.get("version", "0.1")
        self.skills: list[dict[str, Any]] = data.get("skills", [])


class A2AClient:
    """Cliente para comunicarse con agentes remotos via protocolo A2A.

    Obtiene la Agent Card y envía tareas mediante JSON-RPC.
    """

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def get_card(self) -> A2AAgentCard:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base_url}/.well-known/agent.json", headers=headers)
            if resp.status_code == 404:
                resp = await client.get(
                    f"{self._base_url}/a2a/agentCard",
                    headers=headers,
                )
            resp.raise_for_status()
            return A2AAgentCard(resp.json())

    async def send_task(self, message: str, session_id: str = "") -> Any:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "sessionId": session_id or str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"text": message}],
                },
            },
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/a2a",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()


class A2ATool(BaseTool):
    """Tool que envía mensajes a un agente A2A remoto."""

    base_url: str = ""
    api_key: str = ""
    remote_name: str = ""

    def _parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to send to the remote agent",
                }
            },
            "required": ["message"],
        }

    async def run_async(self, **kwargs: Any) -> Any:
        client = A2AClient(self.base_url, self.api_key)
        result = await client.send_task(
            message=kwargs.get("message", ""),
            session_id=kwargs.get("session_id", ""),
        )
        return result
