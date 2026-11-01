"""NAS Agent Development Kit.

Framework async para construir agentes con OpenCode Go, OpenRouter y decisores
TypeSafe Jev u OpenRouter. Inspirado en Google ADK.

Uso básico:
    from nas.adk import Agent, Runner

    from nas.adk.models import ModelRouter

    model = ModelRouter(
        provider="opencode",
        model="deepseek-v4.1-flash",
        api_key="...",
    )
    agent = Agent(name="asistente", model=model, instruction="Eres útil.")
    runner = Runner(app_name="demo", agent=agent)
    async for event in runner.run_async("user1", "sess1", "Hola"):
        print(event.content)
"""

from nas.adk.agents import Decisor
from nas.adk.agents.llm_agent import LlmAgent as Agent
from nas.adk.agents.context import Context
from nas.adk.core.events.event import Event
from nas.adk.runners.runner import Runner
from nas.adk.sessions.in_memory_session_service import InMemorySessionService
from nas.adk.tools.function_tool import FunctionTool
from nas.adk.orchestration.workflow import Workflow, node
from nas.adk.version import __version__

__all__ = [
    "Agent",
    "Context",
    "Decisor",
    "Event",
    "FunctionTool",
    "InMemorySessionService",
    "Runner",
    "Workflow",
    "__version__",
    "node",
]
