"""Programmer settings. Edit this file to configure the example."""

import os
from pathlib import Path

# Modelo y conexión OpenCode; se leen del entorno.
MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")
OPENCODE_API = os.getenv("OPENCODE_API", "auto")
GENERATION_CONFIG = {}  # Ej.: {"reasoning_effort": "low"} si el modelo lo admite.
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go")
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_FALLBACK_MODELS = os.getenv("OPENCODE_FALLBACK_MODELS", "")
REQUEST_TIMEOUT = 120.0
OFFLINE = False  # True: demo que lista archivos, sin modelo ni conexiones.
WORKSPACE = Path(__file__).with_name("workspace")  # Proyecto de ejemplo separado de las claves.
ALLOW_COMMANDS = False  # True autoriza shell con tus permisos; NO es un sandbox.

INSTRUCTION = """Eres un programador útil y conciso.
Lee el código relevante antes de cambiarlo y utiliza las herramientas disponibles.
Delega tareas concretas a subagentes cuando ayude.
Al terminar, resume los cambios y las comprobaciones realizadas."""
