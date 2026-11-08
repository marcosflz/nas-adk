"""API de Anthropic (facturación por API) con la función Python del Runner.

Uso: ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=<id> python ejemplos/claude-gastos/api.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools import consultar_gastos

from nas.adk import Agent, Runner
from nas.adk.models import ModelRouter


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("ANTHROPIC_MODEL"):
        raise RuntimeError("Define ANTHROPIC_API_KEY y ANTHROPIC_MODEL (facturación por API)")
    model = ModelRouter(provider="anthropic")
    agent = Agent(
        name="gastos",
        model=model,
        instruction="Responde con los datos de la herramienta; no inventes cifras.",
        tools=[consultar_gastos],
    )
    try:
        runner = Runner(agent=agent)
        async for event in runner.run_async("yo", "api", "¿Cuánto gasté en 2026-11?"):
            if event.final:
                print(event.content)
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
