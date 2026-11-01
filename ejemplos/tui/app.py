"""Ejemplo de TUI con OpenCode Go.

Configura OPENCODE_API_KEY (y opcionalmente OPENCODE_MODEL) y ejecuta este archivo.
"""

from __future__ import annotations

import os
from pathlib import Path

from nas.adk import Agent
from nas.adk.models import ModelRouter
from nas.adk.tui import AgentApp, TuiConfig


def demo_lookup(query: str) -> str:
    """Devuelve un resultado local de demostración."""
    return f"Resultado de demostración para: {query}"


def suma(a: float, b: float) -> float:
    """Suma dos números y devuelve el resultado."""
    return a + b


def get_db_schema() -> str:
    """Lee el esquema de ejemplo incluido junto a este archivo."""
    return Path(__file__).with_name("schema.dbml").read_text(encoding="utf-8")


ui = TuiConfig(
    title="NAS TUI",
    subtitle="OpenCode",
    content_width=1000,
    css_path=Path(__file__).with_name("nas_tui.css"),
)

model = ModelRouter(
    provider="opencode",
    base_url=os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go"),
    model=os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),
    api=os.getenv("OPENCODE_API", "auto"),
    api_key=os.getenv("OPENCODE_API_KEY", ""),
)

agent = Agent(
    name="NAS-Agent",
    model=model,
    instruction="Responde en español. Usa demo_lookup cuando necesites buscar algo.",
    tools=[demo_lookup, suma, get_db_schema],
)

app = AgentApp(
    agent,
    config=ui,
    session_db="sessions.db",
    tool_policy={
        "demo_lookup": "allow",
        "suma": "allow",
        "get_db_schema": "allow",
    },
)

if __name__ == "__main__":
    app.run()
