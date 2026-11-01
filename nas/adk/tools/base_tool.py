from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel
import jsonschema


class BaseTool(BaseModel, abc.ABC):
    """Clase base para todas las herramientas.

    Una herramienta es una función que el agente puede llamar.
    Se registra en el LLM mediante to_llm_tool() y se ejecuta con run_async().
    """

    name: str = ""
    description: str = ""
    parallel_safe: bool = False

    async def run_with_context(self, ctx: Any, **kwargs: Any) -> Any:
        """Contexto explícito; las tools existentes siguen usando run_async."""
        self.validate_arguments(kwargs)
        return await self.run_async(**kwargs)

    def validate_arguments(self, args: dict[str, Any]) -> None:
        jsonschema.validate(args, self._parameters_schema())

    @abc.abstractmethod
    async def run_async(self, **kwargs: Any) -> Any:
        """Ejecuta la herramienta con los argumentos proporcionados."""
        ...

    def to_llm_tool(self) -> dict[str, Any]:
        """Convierte esta herramienta al formato que espera la API del LLM (OpenAI-compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters_schema(),
            },
        }

    @abc.abstractmethod
    def _parameters_schema(self) -> dict[str, Any]:
        """Devuelve el schema JSON de los parámetros de la herramienta."""
        ...
