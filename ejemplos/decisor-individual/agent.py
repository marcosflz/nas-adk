"""Ejecuta Decisor como agente raíz con TypeSafe Jev.

Uso: TYPESAFE_API_KEY=... python agent.py 'La API devuelve 500'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from nas.adk import Decisor, Runner
from nas.adk.models import ModelRouter


async def main(message: str) -> None:
    api_key = os.getenv("TYPESAFE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Configura TYPESAFE_API_KEY para usar TypeSafe Jev")

    model = ModelRouter(
        provider="typesafe",
        base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
        api_key=api_key,
        model=os.getenv("JEV_MODEL", "jev-latest"),
        timeout=float(os.getenv("JEV_TIMEOUT", "5")),
    )
    decisor = Decisor(
        name="ticket_triage",
        model=model,
        instruction="Clasifica solicitudes de soporte con criterios claros y conservadores.",
        questions={
            "team": {
                "type": "choice",
                "instructions": "¿Qué equipo debe atender principalmente esta solicitud?",
                "criteria": {
                    "technical": "Errores, fallos e integraciones de software",
                    "billing": "Cobros, facturas y suscripciones",
                    "account": "Inicio de sesión y acceso a la cuenta",
                },
            },
            "urgent": {
                "type": "noul",
                "instructions": "¿La solicitud requiere atención urgente?",
            },
        },
    )

    async with model:
        async for event in Runner(agent=decisor).run_async(
            user_id="demo", session_id="ticket-triage", message=message,
        ):
            if event.final:
                if event.status != "completed":
                    raise RuntimeError(event.content or "La decisión no se completó")
                print(json.dumps(json.loads(event.content or "{}"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", default=None, help="Texto de la solicitud")
    args = parser.parse_args()
    asyncio.run(main(args.message or input("Solicitud de soporte: ")))
