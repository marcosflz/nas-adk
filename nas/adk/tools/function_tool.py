from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, PrivateAttr, create_model

from nas.adk.agents.context import Context
from nas.adk.tools.base_tool import BaseTool


class FunctionTool(BaseTool):
    """Schema y validación Pydantic preparados una vez por función."""

    _fn: Callable[..., Any] = PrivateAttr()
    _validator: type[BaseModel] = PrivateAttr()
    _schema: dict[str, Any] = PrivateAttr()
    _context_parameter: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        name: str = "",
        description: str = "",
        parallel_safe: bool = False,
    ) -> None:
        if fn is None:
            raise TypeError("FunctionTool requires a callable")
        super().__init__(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            parallel_safe=parallel_safe,
        )
        self._fn = fn
        hints = get_type_hints(fn, include_extras=True)
        fields: dict[str, Any] = {}
        for pname, param in inspect.signature(fn).parameters.items():
            if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD, param.POSITIONAL_ONLY}:
                raise TypeError(
                    "FunctionTool requires named parameters, not positional-only/*args/**kwargs"
                )
            hint = hints.get(pname, Any)
            if hint is Context:
                if self._context_parameter is not None:
                    raise TypeError("Only one Context parameter is supported")
                self._context_parameter = pname
                continue
            fields[pname] = (hint, ... if param.default is param.empty else param.default)
        self._validator = create_model(
            f"{self.name}Arguments",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        self._schema = self._validator.model_json_schema()

    @property
    def fn(self) -> Callable[..., Any]:
        return self._fn

    def _parameters_schema(self) -> dict[str, Any]:
        return self._schema

    async def _execute(self, kwargs: dict[str, Any], ctx: Context | None) -> Any:
        args = vars(self._validator.model_validate(kwargs)).copy()
        if self._context_parameter:
            if ctx is None:
                raise ValueError("This tool requires an execution Context")
            args[self._context_parameter] = ctx
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**args)
        # ponytail: cancelar el await no detiene el hilo; usar I/O async o timeout propio si importa.
        result = await asyncio.to_thread(self.fn, **args)
        return await result if inspect.isawaitable(result) else result

    async def run_async(self, **kwargs: Any) -> Any:
        return await self._execute(kwargs, None)

    async def run_with_context(self, ctx: Any, **kwargs: Any) -> Any:
        return await self._execute(kwargs, ctx)
