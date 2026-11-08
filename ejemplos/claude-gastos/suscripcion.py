"""Claude Code local con tu suscripción (uso personal), con la misma función Python.

Requisitos: Claude Code instalado y `claude auth login` hecho por ti. Sin ANTHROPIC_API_KEY en
el entorno (si existe, el proveedor se niega a arrancar para no facturar por API).
Uso: python ejemplos/claude-gastos/suscripcion.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools import consultar_gastos

from nas.adk import Agent, Runner
from nas.adk.models import ModelRouter


async def main() -> None:
    model = ModelRouter(provider="claude-code")  # modelo opcional: el configurado en tu CLI
    agent = Agent(
        name="gastos",
        model=model,
        instruction="Responde con los datos de la herramienta; no inventes cifras.",
        tools=[consultar_gastos],
    )
    try:
        runner = Runner(agent=agent)
        async for event in runner.run_async("yo", "suscripcion", "¿Cuánto gasté en 2026-11?"):
            if event.final:
                print(event.content)
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
