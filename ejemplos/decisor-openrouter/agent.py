"""Decisor con OpenRouter y salida JSON Schema.

Uso: OPENROUTER_API_KEY=... OPENROUTER_MODEL=<slug> python agent.py 'La API devuelve 500'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from nas.adk import Decisor, Runner
from nas.adk.models import DecisionOptions, ModelRouter


async def main(message: str) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model_slug = os.getenv("OPENROUTER_MODEL", "")
    if not api_key or not model_slug:
        raise RuntimeError("Configura OPENROUTER_API_KEY y OPENROUTER_MODEL")

    model = ModelRouter(
        provider="openrouter",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        model=model_slug,
        decision=DecisionOptions(strict=True, require_parameters=True, max_retries=2),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        app_title=os.getenv("OPENROUTER_APP_TITLE", "NAS ADK"),
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
            "urgency": {
                "type": "score",
                "instructions": "¿Cuál es la urgencia?",
                "criteria": ["baja", "media", "alta"],
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
