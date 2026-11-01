from __future__ import annotations

from nas.adk import Agent, Runner
from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)


class Step:
    """Paso individual dentro de un plan generado por ReActPlanner."""

    def __init__(
        self, index: int, description: str, status: str = "pending", result: str = ""
    ) -> None:
        self.index = index
        self.description = description
        self.status = status
        self.result = result


class ReActPlanner:
    """Descomposición textual y ejecución de pasos; no observa ni replanifica automáticamente."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def plan(self, goal: str) -> list[Step]:
        svc = InMemorySessionService()
        runner = Runner(app_name="planner", agent=self._agent, session_service=svc)
        prompt = (
            f"Break down this goal into numbered steps (one per line). "
            f"Be specific and actionable.\n\nGoal: {goal}\n\nSteps:"
        )
        final = ""
        async for event in runner.run_async(user_id="planner", session_id="plan", message=prompt):
            if event.final:
                if event.status != "completed":
                    raise RuntimeError(event.content or "Planning did not complete")
                final = event.content or ""
        lines = [ln.strip() for ln in final.split("\n") if ln.strip()]
        steps = []
        for i, line in enumerate(lines):
            desc = line.split(".", 1)[-1].strip() if ". " in line else line
            if desc:
                steps.append(Step(index=i, description=desc))
        return steps or [Step(index=0, description=goal)]

    async def execute_step(self, step: Step, context: str = "") -> str:
        svc = InMemorySessionService()
        runner = Runner(app_name="planner", agent=self._agent, session_service=svc)
        prompt = f"Execute this step:\n{step.description}\nContext: {context}\n\nResult:"
        final = ""
        async for event in runner.run_async(
            user_id="planner", session_id=f"step_{step.index}", message=prompt
        ):
            if event.final:
                if event.status != "completed":
                    raise RuntimeError(event.content or "Step did not complete")
                final = event.content or ""
        return final
