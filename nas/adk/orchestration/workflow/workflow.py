from __future__ import annotations

import asyncio
import json
from graphlib import CycleError, TopologicalSorter
from typing import Any, AsyncGenerator, ClassVar

from pydantic import Field

from nas.adk.agents.context import Context
from nas.adk.models.base_llm import LlmResponse
from nas.adk.orchestration.workflow.base_node import BaseNode
from nas.adk.sessions.session import Session


class Workflow(BaseNode):
    """DAG: uniones de ramas activas; múltiples salidas se agregan por nombre."""

    nodes: dict[str, Any] = Field(default_factory=dict)
    edges: list[tuple[Any, ...]] = Field(default_factory=list)
    max_concurrency: int = Field(default=1, gt=0)
    uses_model: ClassVar[bool] = False

    def resolve_tools(self) -> list[Any]:
        return []

    async def aclose(self) -> None:
        for value in self.nodes.values():
            if callable(getattr(value, "aclose", None)):
                await value.aclose()

    def _resolve(self, value: Any) -> str:
        if hasattr(value, "run") and hasattr(value, "name"):
            if not value.name or value.name in {"START", "END"}:
                raise ValueError("Workflow nodes require a non-reserved name")
            existing = self.nodes.get(value.name)
            if existing is not None and existing is not value:
                raise ValueError(f"Duplicate node name: {value.name}")
            self.nodes[value.name] = value
            return str(value.name)
        return str(value)

    def _graph(self) -> tuple[TopologicalSorter[str], dict[str, list[tuple[str, Any]]]]:
        if not self.edges:
            raise ValueError("Workflow requires edges from START")
        incoming: dict[str, list[tuple[str, Any]]] = {"START": []}
        for edge in self.edges:
            if len(edge) not in {2, 3}:
                raise ValueError("Edges must have two or three elements")
            source, destination = self._resolve(edge[0]), self._resolve(edge[-1])
            route = edge[1] if len(edge) == 3 else None
            if destination == "START" or source == "END":
                raise ValueError("START cannot have predecessors; END cannot have successors")
            for name in (source, destination):
                if name not in {"START", "END"} and (
                    name not in self.nodes or not callable(getattr(self.nodes[name], "run", None))
                ):
                    raise ValueError(f"Unknown workflow node: {name}")
            incoming.setdefault(source, [])
            incoming.setdefault(destination, []).append((source, route))
        reachable = {"START"}
        while True:
            following = {
                name
                for name, sources in incoming.items()
                if any(src in reachable for src, _ in sources)
            }
            if following <= reachable:
                break
            reachable.update(following)
        if set(incoming) - reachable:
            raise ValueError("Every workflow node must be reachable from START")
        sorter = TopologicalSorter(
            {name: {src for src, _ in sources} for name, sources in incoming.items()}
        )
        try:
            sorter.prepare()
        except CycleError as error:
            raise ValueError("Workflow supports DAGs only, not cycles") from error
        return sorter, incoming

    async def run(self, ctx: Any, input_data: Any = None) -> Any:
        sorter, incoming = self._graph()
        results: dict[str, Any] = {"START": input_data}
        used_sources: set[str] = set()

        async def execute(name: str) -> None:
            sources = {
                src: results[src]
                for src, route in incoming[name]
                if src in results and (route is None or results[src] == route)
            }
            if not sources:
                return
            used_sources.update(sources)
            value = next(iter(sources.values())) if len(sources) == 1 else sources
            results[name] = value if name == "END" else await self.nodes[name].run(ctx, value)

        while sorter.is_active():
            ready = list(sorter.get_ready())
            while ready:
                name = ready.pop(0)
                if name == "START":
                    sorter.done(name)
                    continue
                group = [name]
                if getattr(self.nodes.get(name), "parallel_safe", False):
                    while (
                        ready
                        and len(group) < self.max_concurrency
                        and getattr(self.nodes.get(ready[0]), "parallel_safe", False)
                    ):
                        group.append(ready.pop(0))
                tasks = [asyncio.create_task(execute(n)) for n in group]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                sorter.done(*group)
        if "END" in results:
            return results["END"]
        outputs = {
            name: result
            for name, result in results.items()
            if name != "START" and name not in used_sources
        }
        return next(iter(outputs.values())) if len(outputs) == 1 else outputs or None

    def validate_input(self, value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        ctx: Any = None,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        content, _ = await self.run_turn(messages, ctx)
        yield LlmResponse(content=content)

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        ctx: Any = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        result = await self.run(ctx if ctx is not None else Context(Session()), last_user)
        return (result if isinstance(result, str) else json.dumps(result, ensure_ascii=False), [])
