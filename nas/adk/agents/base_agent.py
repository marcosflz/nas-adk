from __future__ import annotations

import abc
import json
from typing import Any, AsyncGenerator

from pydantic import BaseModel, TypeAdapter

from nas.adk.models.base_llm import LlmResponse, ToolCall


class BaseAgent(BaseModel):
    """Agente base del que heredan todos los agentes.

    Atributos:
        name: Nombre del agente.
        instruction: Instrucción/system prompt del agente.
        tools: Lista de herramientas (callables o BaseTool).
        sub_agents: Lista de sub-agentes para Task API.
        input_schema: Schema Pydantic para entrada del Task API.
        output_schema: Schema Pydantic para salida del Task API.
    """

    name: str = ""
    instruction: str = ""
    tools: list[Any] = []
    sub_agents: list[Any] = []
    input_schema: Any = None
    output_schema: Any = None

    def resolve_tools(self) -> list[Any]:
        return []

    def validate_input(self, value: Any) -> str:
        if self.input_schema is not None:
            adapter = TypeAdapter(self.input_schema)
            parsed = (
                adapter.validate_json(value)
                if isinstance(value, str)
                else adapter.validate_python(value)
            )
            return adapter.dump_json(parsed).decode()
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def validate_output(self, response: LlmResponse) -> LlmResponse:
        if self.output_schema is not None and not response.tool_calls:
            adapter = TypeAdapter(self.output_schema)
            parsed = adapter.validate_json(response.content or "")
            response.content = adapter.dump_json(parsed).decode()
        return response

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        ctx: Any = None,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        content, calls = await self.run_turn(messages)
        yield self.validate_output(
            LlmResponse(
                content=content,
                tool_calls=[ToolCall(id=tc["id"], function=tc["function"]) for tc in calls],
            )
        )

    @abc.abstractmethod
    async def run_turn(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Ejecuta un turno del agente: llama al modelo y devuelve contenido + tool_calls.

        Args:
            messages: Historial de mensajes en formato OpenAI.

        Returns:
            Tupla (contenido_texto, lista_de_tool_calls).
        """
        ...
