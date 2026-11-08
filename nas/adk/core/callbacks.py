from __future__ import annotations

import inspect
from typing import Any

from nas.adk.core.events.event import Event
from nas.adk.core.interrupts import ToolDecision
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
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str = "",
        ctx: Any = None,
        agent_name: str = "",
    ) -> ToolDecision | None:
        """Se ejecuta antes de ejecutar una herramienta.

        Puede devolver ``ToolDecision.deny(...)`` o ``ToolDecision.interrupt(...)``.
        Los callbacks antiguos, que solo declaran ``tool_name`` y ``args``, siguen
        funcionando: el Runner solo pasa los argumentos que cada callback acepta.
        """

    async def after_tool_call(
        self, tool_name: str, result: Any, tool_call_id: str = ""
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

    async def run(self, hook: str, **kwargs: Any) -> list[Any]:
        """Ejecuta un hook en todos los callbacks registrados.

        Solo se pasan a cada callback los argumentos que su firma acepta (o todos
        si declara ``**kwargs``), de modo que añadir argumentos al Runner no rompe
        a quien ya tenía callbacks. Devuelve los resultados no nulos.
        """
        results: list[Any] = []
        for cb in self._callbacks:
            method = getattr(cb, hook, None)
            if method is None:
                continue
            result = await method(**_accepted(method, kwargs))
            if result is not None:
                results.append(result)
        return results


def _accepted(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}
