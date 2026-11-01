from __future__ import annotations

from typing import Any, AsyncGenerator, Callable
from contextlib import aclosing

from pydantic import Field, PrivateAttr

from nas.adk.agents.base_agent import BaseAgent
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse
from nas.adk.tools.base_tool import BaseTool
from nas.adk.tools.function_tool import FunctionTool
from nas.adk.tools.sub_agent_tool import SubAgentTool


class LlmAgent(BaseAgent):
    """Agente con modelo de lenguaje. Es la clase principal del ADK.

    Atributos adicionales (además de BaseAgent):
        model: Instancia configurada de BaseLlm.
        tools: Lista de funciones o herramientas.
        sub_agents: Lista de LlmAgent para delegación vía Task API.
    """

    model: BaseLlm
    tools: list[Callable[..., Any] | BaseTool] = []
    sub_agents: list[LlmAgent] = []
    generation_config: dict[str, Any] = Field(default_factory=dict)
    _cached_tools: list[BaseTool] = PrivateAttr(default_factory=list)
    _tools_key: tuple[Any, ...] = PrivateAttr(default=())

    def _resolve_tools(self) -> list[BaseTool]:
        """Convierte callables a FunctionTool y añade tools de sub-agentes."""
        key = tuple(id(t) for t in self.tools) + tuple(
            (id(s), s.name, s.instruction, id(s.input_schema)) for s in self.sub_agents
        )
        if key == self._tools_key:
            return self._cached_tools
        resolved: list[BaseTool] = []
        for t in self.tools:
            if isinstance(t, BaseTool):
                resolved.append(t)
            elif callable(t):
                resolved.append(FunctionTool(t))
        for sub in self.sub_agents:
            resolved.append(self._build_sub_tool(sub))
        self._cached_tools, self._tools_key = resolved, key
        return resolved

    def resolve_tools(self) -> list[BaseTool]:
        return self._resolve_tools()

    async def run(self, ctx: Any, input_data: Any = None) -> str:
        result: str = await ctx.run_agent(self, input_data)
        return result

    def prepare_request(
        self,
        messages: list[dict[str, Any]],
        ctx: Any = None,
    ) -> LlmRequest:
        config = dict(self.generation_config)
        forbidden = config.keys() - LlmRequest.model_fields.keys()
        if forbidden or config.keys() & {"model", "messages", "tools", "stream", "session_id"}:
            raise ValueError("Invalid generation_config options")
        if self.output_schema is not None:
            config["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.name or "response",
                    "schema": self.output_schema.model_json_schema(),
                },
            }
        return LlmRequest(
            model=self.model.model,
            session_id=ctx.session.id if ctx is not None else None,
            messages=messages,
            tools=[t.to_llm_tool() for t in self.resolve_tools()] or None,
            **config,
        )

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        ctx: Any = None,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        request = self.prepare_request(messages, ctx)
        model = self.model
        if stream and getattr(model, "supports_streaming", False) is True:
            async with aclosing(model.generate_stream_async(request)) as responses:
                async for response in responses:
                    yield response if response.type == "delta" else self.validate_output(response)
        else:
            yield self.validate_output(await model.generate_content_async(request))

    async def aclose(self) -> None:
        for sub in self.sub_agents:
            await sub.aclose()

    async def aclose_models(self) -> None:
        """Cierra los modelos de este árbol; el llamador decide cuándo es su dueño."""
        seen: set[int] = set()

        async def close(agent: LlmAgent) -> None:
            if id(agent.model) not in seen:
                seen.add(id(agent.model))
                await agent.model.aclose()
            for child in agent.sub_agents:
                await close(child)

        await close(self)

    def _build_sub_tool(self, sub: LlmAgent) -> SubAgentTool:
        """Crea un SubAgentTool para delegar tareas a un sub-agente."""
        instr = (sub.instruction or "")[:120]
        return SubAgentTool(
            name=f"request_task_{sub.name}",
            description=f"Delegate to {sub.name}: {instr}",
            sub_agent=sub,
        )

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Ejecuta un turno: resuelve modelo, construye request y devuelve respuesta.

        Args:
            messages: Historial en formato OpenAI.

        Returns:
            (contenido_texto, lista_de_tool_calls_en_formato_dict).
        """
        async for response in self.generate_async(messages):
            return response.content, response.tool_calls_dicts()
        raise RuntimeError("Agent did not produce a response")
