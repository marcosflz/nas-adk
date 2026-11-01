from __future__ import annotations

from typing import Any

from nas.adk.tools.base_tool import BaseTool


class SubAgentTool(BaseTool):
    """Tool que representa un sub-agente para la Task API.

    El Runner intercepta las llamadas a SubAgentTool y ejecuta
    el sub-agente en lugar de una herramienta normal.
    """

    sub_agent: Any = None

    def _parameters_schema(self) -> dict[str, Any]:
        sub = self.sub_agent
        if sub is not None and sub.input_schema is not None:
            return dict(sub.input_schema.model_json_schema())
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description or input",
                }
            },
            "required": ["task"],
        }

    async def run_async(self, **kwargs: Any) -> Any:
        raise RuntimeError("SubAgentTool must be handled by Runner")
