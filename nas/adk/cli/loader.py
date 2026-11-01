from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import cast

from nas.adk import Agent


def load_agent(agent_path: str) -> Agent:
    """Carga un agente desde un archivo Python.

    Busca un objeto 'root_agent' (instancia de Agent) en el módulo.
    Carga .env del directorio del agente si existe.

    Args:
        agent_path: Ruta al directorio del agente o al archivo agent.py.

    Returns:
        Instancia del agente cargada.
    """
    path = Path(agent_path).resolve()
    if path.is_dir():
        env_path = path / ".env"
        path = path / "agent.py"
    else:
        env_path = path.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")
    spec = importlib.util.spec_from_file_location("agent_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = mod
    spec.loader.exec_module(mod)
    agent = cast(Agent | None, getattr(mod, "root_agent", None))
    if agent is None:
        raise ValueError(
            f"No `root_agent` found in {path}."
        )
    return agent
