from __future__ import annotations

import inspect
import asyncio
from typing import Any, Callable

from nas.adk.orchestration.workflow.base_node import BaseNode


def node(fn: Callable[..., Any]) -> FunctionNode:
    """Decorador que convierte una función en un FunctionNode para Workflow."""
    return FunctionNode(fn=fn, name=fn.__name__)


class FunctionNode(BaseNode):
    """Nodo de Workflow que envuelve una función Python.

    La función recibe (ctx, input_data) y devuelve un resultado
    que se pasa al siguiente nodo del grafo.
    """

    fn: Callable[..., Any] | None = None

    async def run(self, ctx: Any, input_data: Any = None) -> Any:
        if self.fn is None:
            raise RuntimeError("FunctionNode has no callable")
        sig = inspect.signature(self.fn)
        params = list(sig.parameters.keys())
        kwargs: dict[str, Any] = {}
        if params and params[0] == "ctx":
            kwargs["ctx"] = ctx
        if len(params) > 1:
            kwargs[params[1]] = input_data
        elif params and params[0] != "ctx":
            kwargs[params[0]] = input_data
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        result = await asyncio.to_thread(self.fn, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
