from __future__ import annotations

import abc
import json
from typing import Any, AsyncGenerator, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """Llamada a herramienta solicitada por el modelo."""

    id: str = ""
    function: dict[str, Any] = Field(default_factory=dict)


class LlmRequest(BaseModel):
    """Petición al modelo: mensajes + herramientas disponibles."""

    model_config = ConfigDict(extra="forbid")

    model: str = ""
    session_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None


class LlmResponse(BaseModel):
    """Respuesta del modelo: texto generado + tool_calls."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    type: Literal["delta", "response"] = "response"
    usage: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    response_items: list[dict[str, Any]] = Field(default_factory=list)

    def tool_calls_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.get("name", ""),
                    "arguments": (
                        tc.function.get("arguments", "{}")
                        if isinstance(tc.function.get("arguments"), str)
                        else json.dumps(tc.function.get("arguments", {}))
                    ),
                },
            }
            for tc in self.tool_calls
        ]


class BaseLlm(BaseModel, abc.ABC):
    """Contrato async compartido por modelos y adaptadores de prueba."""

    model: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    supports_streaming: ClassVar[bool] = False

    async def aclose(self) -> None:
        """Libera recursos propios; las implementaciones sin recursos no hacen nada."""

    async def __aenter__(self) -> BaseLlm:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    @abc.abstractmethod
    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        """Llama al modelo de forma no-streaming.

        Args:
            request: Mensajes, herramientas y configuración.

        Returns:
            Respuesta completa del modelo.
        """
        ...

    async def generate_stream_async(self, request: LlmRequest) -> AsyncGenerator[LlmResponse, None]:
        """Llama al modelo en modo streaming.

        Por defecto delega a generate_content_async. Los proveedores
        que soporten streaming deben sobrescribir este método.
        """
        yield await self.generate_content_async(request)

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        """Evalúa preguntas estructuradas si el modelo ofrece esa capacidad."""
        raise NotImplementedError(f"{type(self).__name__} does not support structured decisions")
