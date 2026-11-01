from __future__ import annotations

from typing import Any

from nas.adk.core.events.event import Event
from nas.adk.models.base_llm import LlmResponse


class BaseCallback:
    """Hook base para observar el ciclo de vida de un agente.

    Los métodos se ejecutan en puntos clave del Runner.
    Hereda de esta clase y sobrescribe los métodos que necesites.
    """

    async def before_model_call(
        self, agent_name: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> None:
        """Se ejecuta antes de llamar al modelo."""

    async def after_model_call(
        self, agent_name: str, response: LlmResponse
    ) -> None:
        """Se ejecuta después de recibir respuesta del modelo."""

    async def before_tool_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> None:
        """Se ejecuta antes de ejecutar una herramienta."""

    async def after_tool_call(
        self, tool_name: str, result: Any
    ) -> None:
        """Se ejecuta después de ejecutar una herramienta."""

    async def on_event(self, event: Event) -> None:
        """Se ejecuta cuando se genera un evento."""

    async def on_error(self, error: Exception) -> None:
        """Se ejecuta cuando ocurre un error."""


class CallbackManager:
    """Gestiona una lista de callbacks y los ejecuta en cada hook."""

    def __init__(self) -> None:
        self._callbacks: list[BaseCallback] = []

    @property
    def callbacks(self) -> list[BaseCallback]:
        """Devuelve una copia de la lista de callbacks."""
        return list(self._callbacks)

    def add(self, callback: BaseCallback) -> None:
        """Registra un callback."""
        self._callbacks.append(callback)

    async def run(self, hook: str, **kwargs: Any) -> None:
        """Ejecuta un hook en todos los callbacks registrados."""
        for cb in self._callbacks:
            method = getattr(cb, hook, None)
            if method is not None:
                await method(**kwargs)
