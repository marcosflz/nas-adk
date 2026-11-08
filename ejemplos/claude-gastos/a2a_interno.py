"""Dos agentes propios hablando por A2A en local, ambos con Claude Code (suscripción).

Un agente "analista" se sirve por A2A (127.0.0.1, con clave) y un "orquestador" lo consulta como
herramienta. Es para automatizaciones internas propias: ``allow_personal_use_only=True`` solo
admite loopback y exige clave; no abras este puerto a otras personas.

Uso:
  python ejemplos/claude-gastos/a2a_interno.py --offline   # CLI falso, sin gastar cuota
  python ejemplos/claude-gastos/a2a_interno.py             # Claude Code real (suscripción)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

import uvicorn
from tools import consultar_gastos

from nas.adk import Agent, Runner
from nas.adk.models.claude_code import ClaudeCodeLlm
from nas.adk.protocols.a2a.client import A2ATool
from nas.adk.protocols.a2a.server import create_a2a_app

HOST = "127.0.0.1"


def free_port() -> int:
    if os.getenv("A2A_PORT"):
        return int(os.environ["A2A_PORT"])
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])



def offline_models() -> tuple[ClaudeCodeLlm, ClaudeCodeLlm]:
    """ClaudeCodeLlm real con un transporte de procesos simulado (no ejecuta `claude`)."""
    from tests.claude_code_fakes import INIT, FakeHandle, FakeTransport, calls, final

    analista = FakeTransport(
        [
            FakeHandle([INIT, calls(("consultar_gastos", {"mes": "2026-11"}))]),
            FakeHandle([INIT, final("En 2026-11 gastaste 987.25 EUR, menos que en octubre.")]),
        ]
    )
    orquestador = FakeTransport(
        [
            FakeHandle([INIT, calls(("consultar_analista", {"message": "Resume 2026-11"}))]),
            FakeHandle([INIT, final("Informe listo: el analista confirma un mes más contenido.")]),
        ]
    )
    return ClaudeCodeLlm(transport=analista), ClaudeCodeLlm(transport=orquestador)


async def main(offline: bool) -> None:
    port = free_port()
    key = secrets.token_urlsafe(24)  # clave efímera solo para esta ejecución
    if offline:
        analista_model, orquestador_model = offline_models()
    else:
        analista_model, orquestador_model = ClaudeCodeLlm(), ClaudeCodeLlm()

    analista = Agent(
        name="analista",
        model=analista_model,
        instruction="Analiza gastos usando la herramienta; no inventes cifras.",
        tools=[consultar_gastos],
    )
    app = create_a2a_app(
        analista, api_key=key, host=HOST, allow_personal_use_only=True
    )
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        if serving.done():
            raise RuntimeError(f"El servidor A2A no pudo arrancar en {HOST}:{port}")
        await asyncio.sleep(0.05)

    consultar_analista = A2ATool(
        name="consultar_analista",
        description="Pregunta al agente analista de gastos y devuelve su respuesta.",
        base_url=f"http://{HOST}:{port}",
        api_key=key,
        remote_name="analista",
    )
    orquestador = Agent(
        name="orquestador",
        model=orquestador_model,
        instruction="Delega el análisis en el analista y resume su respuesta.",
        tools=[consultar_analista],
    )
    try:
        runner = Runner(agent=orquestador, timeout=600)
        async for event in runner.run_async("yo", "interno", "Dame un informe de gastos de 2026-11."):
            if event.final:
                print(event.content)
    finally:
        server.should_exit = True
        await serving
        await orquestador_model.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args: Any = parser.parse_args()
    asyncio.run(main(args.offline))
