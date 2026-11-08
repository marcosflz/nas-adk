"""Entorno, argumentos y comprobaciones de aislamiento/autenticación de Claude Code.

Todo el filtrado de variables se hace sobre una copia local: ``os.environ`` no se modifica.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from nas.adk.models.errors import (
    LlmAuthError,
    LlmBillingModeError,
    LlmIsolationError,
    LlmProcessError,
)

MIN_VERSION = (2, 1, 259)
TESTED_VERSION = "2.1.319"

# Flags imprescindibles (aislamiento + salida estructurada). Si falta alguna, se rechaza el CLI.
REQUIRED_FLAGS = (
    "--tools",
    "--disallowedTools",
    "--strict-mcp-config",
    "--restricted",
    "--safe-mode",
    "--disable-slash-commands",
    "--permission-mode",
    "--permission-prompts",
    "--no-session-persistence",
    "--json-schema",
    "--output-format",
    "--system-prompt",
)

# Variables que pueden cambiar el backend o la credencial (y por tanto la facturación).
BILLING_EXACT = frozenset(
    {
        "AWS_BEARER_TOKEN_BEDROCK",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    }
)
BILLING_PREFIXES = ("ANTHROPIC_",)
BILLING_VAR_EXTRA = ("CLAUDE_CODE_USE_",)

# Desactiva funciones del CLI que no deben intervenir en una inferencia aislada.
HARDENING_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_DISABLE_CRON": "1",
    "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
    "CLAUDE_CODE_DISABLE_ARTIFACT": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
}

_SECRET_RE = re.compile(r"(sk-ant-[A-Za-z0-9_\-]+|eyJ[A-Za-z0-9_\-\.]{20,}|[A-Za-z0-9_\-]{40,})")


def redact(text: str, limit: int = 500) -> str:
    """Elimina posibles secretos y acota la longitud de un fragmento de salida."""
    return _SECRET_RE.sub("[redacted]", text).strip()[-limit:]


def is_billing_var(name: str) -> bool:
    return (
        name in BILLING_EXACT
        or name.startswith(BILLING_PREFIXES)
        or name.startswith(BILLING_VAR_EXTRA)
    )


def build_child_env(parent: Mapping[str, str], *, policy: str) -> dict[str, str]:
    """Copia del entorno para el proceso hijo.

    ``policy="reject"`` (por defecto): si hay variables de backend/credencial de API, se lanza
    ``LlmBillingModeError`` (solo se citan los nombres). ``"scrub"``: se eliminan de la copia.
    """
    present = sorted(name for name in parent if is_billing_var(name) and parent[name])
    if present and policy != "scrub":
        raise LlmBillingModeError(
            "Refusing to start Claude Code in subscription mode: these variables could route "
            f"requests to a billed backend: {', '.join(present)}. Unset them, or set "
            "CLAUDE_CODE_ENV_POLICY=scrub to drop them from the child process only."
        )
    env = {k: v for k, v in parent.items() if not is_billing_var(k)}
    env.update(HARDENING_ENV)
    return env


def parse_version(text: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        raise LlmProcessError(f"Cannot parse the Claude Code version from {text.strip()[:80]!r}")
    return int(match[1]), int(match[2]), int(match[3])


def check_cli(version_output: str, help_output: str) -> tuple[int, int, int]:
    version = parse_version(version_output)
    if version < MIN_VERSION:
        raise LlmProcessError(
            f"Claude Code {'.'.join(map(str, version))} is too old; "
            f"{'.'.join(map(str, MIN_VERSION))} or newer is required (tested: {TESTED_VERSION})"
        )
    missing = [flag for flag in REQUIRED_FLAGS if flag not in help_output]
    if missing:
        raise LlmProcessError(
            f"This Claude Code build lacks flags required for isolation: {', '.join(missing)}"
        )
    return version


def check_auth_status(stdout: str, exit_code: int) -> None:
    """Valida la salida de ``claude auth status --json``: solo suscripción de claude.ai."""
    try:
        status: Any = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise LlmProcessError("`claude auth status` did not return JSON") from error
    if exit_code != 0 or not isinstance(status, dict) or not status.get("loggedIn"):
        raise LlmAuthError(
            "Claude Code is not logged in. Run `claude auth login` yourself with your "
            "subscription; NAS never starts a login."
        )
    method = status.get("authMethod")
    provider = status.get("apiProvider")
    if provider != "firstParty" or method not in {"claude.ai", "oauth_token"}:
        raise LlmBillingModeError(
            f"Claude Code is authenticated with authMethod={method!r}, apiProvider={provider!r}; "
            "subscription mode requires a claude.ai login (authMethod 'claude.ai' or "
            "'oauth_token') on the first-party provider."
        )


def check_init_event(event: Mapping[str, Any]) -> None:
    """Comprobación detectiva del evento ``system/init``."""
    problems: list[str] = []
    if event.get("apiKeySource") != "none":
        problems.append(f"apiKeySource={event.get('apiKeySource')!r} (expected 'none')")
    tools = [str(t) for t in event.get("tools") or []]
    extra = [t for t in tools if t != "StructuredOutput"]
    if extra:
        problems.append(f"unexpected tools enabled: {extra}")
    if event.get("mcp_servers"):
        problems.append("MCP servers are connected")
    # Los plugins integrados del propio CLI (path "builtin") no ejecutan nada sin herramientas.
    external = [
        p for p in event.get("plugins") or [] if not (isinstance(p, dict) and p.get("path") == "builtin")
    ]
    if external:
        problems.append("non-builtin plugins are loaded")
    if problems:
        raise LlmIsolationError("Claude Code isolation check failed: " + "; ".join(problems))


def build_argv(
    executable: str,
    *,
    system_prompt: str,
    schema_json: str,
    model: str | None,
    effort: str | None,
    optional_flags: set[str],
) -> list[str]:
    """Argumentos fijos y revisados; el contenido del usuario va siempre por stdin."""
    argv = [
        executable,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--tools",
        "",
        "--disallowedTools",
        "mcp__*",
        "--strict-mcp-config",
        "--restricted",
        "--safe-mode",
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
        "--permission-prompts",
        "none",
        "--system-prompt",
        system_prompt,
        "--json-schema",
        schema_json,
    ]
    if "--max-turns" in optional_flags:
        argv += ["--max-turns", "4"]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    return argv
