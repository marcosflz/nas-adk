from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseNode(BaseModel):
    """Nodo base para grafos de Workflow.

    Cualquier elemento ejecutable en un grafo (agente, función, workflow anidado)
    hereda de BaseNode e implementa run().
    """

    name: str = ""
    parallel_safe: bool = False

    async def run(self, ctx: Any, input_data: Any = None) -> Any:
        """Ejecuta el nodo con el contexto y datos de entrada dados."""
        raise NotImplementedError
