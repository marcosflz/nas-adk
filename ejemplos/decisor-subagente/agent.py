"""Decisor TypeSafe Jev como subagente de un chat OpenCode.

Uso: TYPESAFE_API_KEY=... OPENCODE_API_KEY=... python agent.py 'La API devuelve 500'
"""

from __future__ import annotations

import argparse
import asyncio
import os

from pydantic import BaseModel

from nas.adk import Agent, Decisor, Runner
from nas.adk.models import ModelRouter


class SupportRequest(BaseModel):
    message: str


async def main(message: str) -> None:
    typesafe_key = os.getenv("TYPESAFE_API_KEY", "")
    opencode_key = os.getenv("OPENCODE_API_KEY", "")
    if not typesafe_key or not opencode_key:
        raise RuntimeError("Configura TYPESAFE_API_KEY y OPENCODE_API_KEY")

    decision_model = ModelRouter(
        provider="typesafe",
        base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
        api_key=typesafe_key,
        model=os.getenv("JEV_MODEL", "jev-latest"),
    )
    chat_model = ModelRouter(
        provider="opencode",
        base_url=os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go"),
        api_key=opencode_key,
        model=os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),
        api=os.getenv("OPENCODE_API", "auto"),
    )
    decisor = Decisor(
        name="triage",
        model=decision_model,
        input_schema=SupportRequest,
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
    assistant = Agent(
        name="support_assistant",
        model=chat_model,
        instruction=(
            "Para clasificar la solicitud, delega siempre en request_task_triage y pasa "
            "el texto completo en el campo message. Usa las respuestas estructuradas "
            "para explicar brevemente el equipo y la urgencia."
        ),
        sub_agents=[decisor],
    )

    async with decision_model, chat_model:
        async for event in Runner(agent=assistant).run_async(
            user_id="demo", session_id="support-triage", message=message,
        ):
            if event.role == "tool":
                print(f"Resultado de Decisor: {event.content}")
            if event.final:
                if event.status != "completed":
                    raise RuntimeError(event.content or "El agente no se completó")
                print(f"Asistente: {event.content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", default=None, help="Texto de la solicitud")
    args = parser.parse_args()
    asyncio.run(main(args.message or input("Solicitud de soporte: ")))
