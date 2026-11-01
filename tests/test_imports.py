from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_public_imports() -> None:
    from nas.adk import Agent, Decisor, FunctionTool, Runner, __version__
    from nas.adk.agents import Agent as PublicAgent
    from nas.adk.agents import BaseAgent, Context, Decisor as PublicDecisor, LlmAgent
    from nas.adk.models import (
        BaseLlm,
        JevLlm,
        ModelRouter,
        OpenCodeLlm,
        OpenRouterLlm,
        TypesafeLlm,
    )
    from nas.adk.runners import Runner as PublicRunner
    from nas.adk.sessions import SQLiteSessionService
    from nas.adk.tools import BaseTool, SubAgentTool
    from nas.adk.tools import FunctionTool as PublicFunctionTool
    from nas.adk.version import __version__ as module_version

    assert Agent is PublicAgent is LlmAgent
    assert Decisor is PublicDecisor
    assert Runner is PublicRunner
    assert FunctionTool is PublicFunctionTool
    assert issubclass(Agent, BaseAgent)
    assert issubclass(ModelRouter, BaseLlm)
    assert issubclass(OpenCodeLlm, BaseLlm)
    assert issubclass(OpenRouterLlm, BaseLlm)
    assert issubclass(TypesafeLlm, BaseLlm)
    assert JevLlm is TypesafeLlm
    assert issubclass(SubAgentTool, BaseTool)
    assert callable(Context) and callable(SQLiteSessionService)
    assert __version__ == module_version


def test_package_imports_without_optional_sdk(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "from nas.adk import Agent, Decisor; "
            "from nas.adk.models import ModelRouter, OpenCodeLlm, OpenRouterLlm, TypesafeLlm; "
            "assert callable(Agent) and callable(Decisor) and callable(ModelRouter)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert root.is_dir()
