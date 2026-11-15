"""Proveedor ``claude-code``: adaptador de proceso local sobre el CLI oficial de Claude Code.

No es un cliente HTTP de Anthropic. El usuario instala y autentica Claude Code por separado (con
su suscripción); NAS lo invoca de forma aislada, sin herramientas propias del CLI, y solo usa su
salida estructurada para *decidir*. Las funciones Python las ejecuta el Runner de NAS.

Uso pensado para el titular de la suscripción, en local y sin compartir (véase el README).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from typing import Any, ClassVar, Literal

from pydantic import Field, PrivateAttr, SecretStr

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ModelCapabilities
from nas.adk.models.catalog import ModelInfo
from nas.adk.models.claude_code import isolation
from nas.adk.models.claude_code import protocol as proto
from nas.adk.models.claude_code.transport import (
    AsyncioProcessTransport,
    ProcessTransport,
    capture,
)
from nas.adk.models.decision import (
    build_structured_decision_messages,
    normalize_decision_usage,
    parse_decision_answers,
    structured_decision_schema,
)
from nas.adk.models.errors import (
    LlmAuthError,
    LlmProcessError,
    LlmProtocolError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmRequestError,
    LlmTimeoutError,
    LlmUnsupportedError,
)
from nas.adk.models.openai_compat import env_float, env_int

_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


class ClaudeCodeLlm(BaseLlm):
    """Claude Code local (suscripción) con tool calling emulado mediante salida estructurada."""

    provider: Literal["claude-code"] = "claude-code"
    model: str = Field(default_factory=lambda: os.getenv("CLAUDE_CODE_MODEL", ""))
    executable: str = Field(default_factory=lambda: _env_str("CLAUDE_CODE_BIN", "claude"))
    timeout: float = Field(default_factory=lambda: env_float("CLAUDE_CODE_TIMEOUT", 300.0), gt=0)
    max_concurrency: int = Field(
        default_factory=lambda: env_int("CLAUDE_CODE_MAX_CONCURRENCY", 1), ge=1
    )
    env_policy: Literal["reject", "scrub"] = Field(
        default_factory=lambda: _env_str("CLAUDE_CODE_ENV_POLICY", "reject")  # type: ignore[arg-type]
    )
    max_input_bytes: int = Field(
        default_factory=lambda: env_int("CLAUDE_CODE_MAX_INPUT_BYTES", 9_000_000), gt=0
    )
    max_tool_calls: int = Field(
        default_factory=lambda: env_int("CLAUDE_CODE_MAX_TOOL_CALLS", 8), ge=1
    )
    protocol_repairs: int = Field(
        default_factory=lambda: env_int("CLAUDE_CODE_PROTOCOL_REPAIRS", 1), ge=0
    )
    auth_check_ttl: float = Field(
        default_factory=lambda: env_float("CLAUDE_CODE_AUTH_CHECK_TTL", 300.0), ge=0
    )
    oauth_token: SecretStr | None = Field(default=None, repr=False)
    """Token de suscripción (``claude setup-token``) solo para los procesos de esta instancia.

    Se inyecta como ``CLAUDE_CODE_OAUTH_TOKEN`` en la copia del entorno del proceso hijo, nunca
    en ``os.environ``: así varias instancias con cuentas distintas pueden convivir en el mismo
    proceso sin pisarse. Sin él, el CLI usa su login propio o la variable del entorno.
    """
    transport: Any = None

    supports_streaming: ClassVar[bool] = False
    capabilities: ClassVar[ModelCapabilities | None] = ModelCapabilities(
        tool_calling="emulated",
        structured_output="emulated",
        decisions="structured",
        text_streaming=False,
        tool_argument_streaming=False,
        usage="partial",
        personal_use_only=True,
    )

    _semaphore: asyncio.Semaphore | None = PrivateAttr(default=None)
    _ready: set[str] | None = PrivateAttr(default=None)  # flags opcionales disponibles
    _auth_ok_until: float = PrivateAttr(default=0.0)

    # ------------------------------------------------------------------ preparación

    def _child_env(self) -> dict[str, str]:
        env = isolation.build_child_env(os.environ, policy=self.env_policy)
        if self.oauth_token is not None:
            env[isolation.OAUTH_TOKEN_VAR] = self.oauth_token.get_secret_value()
        return env

    def _transport(self) -> ProcessTransport:
        if self.transport is None:
            self.transport = AsyncioProcessTransport()
        return self.transport  # type: ignore[no-any-return]

    async def _prepare(self, env: dict[str, str]) -> tuple[str, set[str]]:
        transport = self._transport()
        path = transport.which(self.executable)
        if not path:
            raise LlmProcessError(
                f"Claude Code executable {self.executable!r} was not found. Install it and log in "
                "yourself, or set CLAUDE_CODE_BIN."
            )
        with tempfile.TemporaryDirectory(prefix="nas-cc-") as cwd:
            if self._ready is None:
                code, out, _ = await capture(
                    transport, [path, "--version"], env=env, cwd=cwd, timeout=30
                )
                if code != 0:
                    raise LlmProcessError("`claude --version` failed")
                code, help_out, _ = await capture(
                    transport, [path, "--help"], env=env, cwd=cwd, timeout=30
                )
                isolation.check_cli(out, help_out)
                self._ready = {f for f in ("--max-turns", "--effort") if f in help_out}
            if time.monotonic() >= self._auth_ok_until:
                code, out, _ = await capture(
                    transport, [path, "auth", "status", "--json"], env=env, cwd=cwd, timeout=30
                )
                isolation.check_auth_status(out, code)
                self._auth_ok_until = time.monotonic() + self.auth_check_ttl
        return path, self._ready

    # ------------------------------------------------------------------ catálogo

    async def list_models(self, timeout: float = 60.0) -> list[ModelInfo]:
        """Modelos que la suscripción autenticada puede usar, según el propio CLI.

        Usa el protocolo de control oficial (``initialize`` sobre ``--input-format stream-json``,
        lo mismo que ``supportedModels()`` del Agent SDK). No envía ningún mensaje de usuario, así
        que no hace inferencia ni gasta cuota.
        """
        env = self._child_env()
        path, _ = await self._prepare(env)
        argv = isolation.build_control_argv(path)
        request_id = f"nas-models-{time.monotonic_ns()}"
        request = {"type": "control_request", "request_id": request_id, "request": {"subtype": "initialize"}}
        with tempfile.TemporaryDirectory(prefix="nas-cc-") as cwd:
            handle = await self._transport().spawn(argv, env=env, cwd=cwd)
            try:
                async def _read() -> dict[str, Any] | None:
                    # El CLI responde a `initialize` aunque stdin se cierre justo después.
                    await handle.write_stdin((json.dumps(request) + "\n").encode())
                    async for raw in handle.stdout_lines():
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if (
                            isinstance(event, dict)
                            and event.get("type") == "control_response"
                            and (event.get("response") or {}).get("request_id") == request_id
                        ):
                            return event["response"]  # type: ignore[no-any-return]
                    return None

                try:
                    response = await asyncio.wait_for(_read(), timeout)
                except asyncio.TimeoutError as error:
                    raise LlmTimeoutError("Claude Code did not list its models in time", cause=error) from error
            finally:
                await handle.terminate()
        if not response or response.get("subtype") != "success":
            detail = isolation.redact(str((response or {}).get("error") or handle.stderr_tail()))
            raise LlmProcessError(f"Claude Code did not return its model list: {detail}")
        return parse_models((response.get("response") or {}).get("models"))

    # ------------------------------------------------------------------ ejecución

    async def _run_cli(
        self, stdin_text: str, schema: dict[str, Any], effort: str | None
    ) -> dict[str, Any]:
        """Una invocación aislada. Devuelve el mensaje ``result`` (con metadatos añadidos)."""
        data = stdin_text.encode("utf-8")
        if len(data) > self.max_input_bytes:
            raise LlmRequestError(
                f"Conversation is {len(data)} bytes, above CLAUDE_CODE_MAX_INPUT_BYTES="
                f"{self.max_input_bytes}; shrink the context explicitly (no silent truncation)"
            )
        schema_json = json.dumps(schema, separators=(",", ":"))
        if len(schema_json.encode()) > isolation_arg_limit():
            raise LlmRequestError("Output schema is too large to pass to Claude Code")
        env = self._child_env()
        path, optional = await self._prepare(env)
        argv = isolation.build_argv(
            path,
            system_prompt=proto.SYSTEM_PROMPT,
            schema_json=schema_json,
            model=self.model or None,
            effort=effort if "--effort" in optional else None,
            optional_flags=optional,
        )
        async with self._get_semaphore():
            try:
                return await asyncio.wait_for(self._execute(argv, env, data), self.timeout)
            except asyncio.TimeoutError as error:
                raise LlmTimeoutError(
                    f"Claude Code did not finish within {self.timeout:g}s", cause=error
                ) from error

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def _execute(self, argv: list[str], env: dict[str, str], data: bytes) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="nas-cc-") as cwd:
            handle = await self._transport().spawn(argv, env=env, cwd=cwd)
            rate_rejected: dict[str, Any] | None = None
            assistant_error: str | None = None
            result: dict[str, Any] | None = None
            try:
                await handle.write_stdin(data)
                async for raw in handle.stdout_lines():
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # ruido no JSON: el envelope válido llega en `result`
                    if not isinstance(event, dict):
                        continue
                    kind, sub = event.get("type"), event.get("subtype")
                    if kind == "system" and sub == "init":
                        isolation.check_init_event(event)
                    elif kind == "rate_limit_event":
                        info = event.get("rate_limit_info") or {}
                        if info.get("status") == "rejected":
                            rate_rejected = info
                    elif kind == "assistant" and event.get("error"):
                        assistant_error = str(event["error"])
                    elif kind == "result":
                        result = event
                        break
                code = await handle.wait() if result is None else 0
                tail = isolation.redact(handle.stderr_tail())
            finally:
                await handle.terminate()
        if result is None:
            raise LlmProcessError(
                f"Claude Code exited with code {code} without a result"
                + (f": {tail}" if tail else "")
            )
        result["_rate_rejected"] = rate_rejected
        result["_assistant_error"] = assistant_error
        return result

    @staticmethod
    def _raise_for_result(result: dict[str, Any]) -> None:
        rejected = result.get("_rate_rejected")
        aerr = result.get("_assistant_error")
        subtype = result.get("subtype")
        text = str(result.get("result") or "")
        if rejected is not None:
            resets = rejected.get("resetsAt")
            raise LlmQuotaError(
                "Claude subscription usage limit reached"
                + (f" (resets at {resets})" if resets else "")
                + ". NAS does not fall back to a paid API.",
                fallback_allowed=False,
            )
        if aerr in {"authentication_failed", "oauth_org_not_allowed", "account_on_hold"}:
            raise LlmAuthError(f"Claude Code authentication failed ({aerr}); run `claude auth login`")
        if aerr == "rate_limit":
            raise LlmRateLimitError("Claude Code reported a rate limit", fallback_allowed=False)
        if aerr == "billing_error":
            raise LlmQuotaError("Claude Code reported a billing/usage error", fallback_allowed=False)
        if subtype in {"error_max_structured_output_retries", "error_max_turns"}:
            raise LlmProtocolError(f"Claude Code could not produce valid output ({subtype})")
        if result.get("is_error") or subtype != "success":
            if "login" in text.lower() or "log in" in text.lower():
                raise LlmAuthError("Claude Code is not logged in; run `claude auth login`")
            detail = "; ".join(str(e) for e in result.get("errors") or []) or text
            raise LlmProcessError(
                f"Claude Code failed ({subtype}): {isolation.redact(detail)}"
            )
        if "structured_output" not in result or result["structured_output"] is None:
            raise LlmProtocolError("Claude Code returned success without structured_output")

    @staticmethod
    def _usage(result: dict[str, Any]) -> dict[str, Any]:
        usage = result.get("usage")
        if not isinstance(usage, dict):
            return {}
        out: dict[str, Any] = {}
        inp = usage.get("input_tokens")
        outp = usage.get("output_tokens")
        if isinstance(inp, int) and not isinstance(inp, bool):
            extra = sum(
                v
                for k in ("cache_read_input_tokens", "cache_creation_input_tokens")
                if isinstance(v := usage.get(k), int)
            )
            out["prompt_tokens"] = out["input_tokens"] = inp + extra
        if isinstance(outp, int) and not isinstance(outp, bool):
            out["completion_tokens"] = out["output_tokens"] = outp
        if "prompt_tokens" in out and "completion_tokens" in out:
            out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        return out

    def _metrics(self, result: dict[str, Any], mode: str, repairs: int, ignored: list[str]) -> dict[str, Any]:
        return {
            "provider": "claude-code",
            "model": self.model or None,
            "claude_code_schema_mode": mode,
            "protocol_repairs": repairs,
            "ignored_parameters": ignored,
            "session_id": result.get("session_id"),
            "cost_usd": None,
        }

    # ------------------------------------------------------------------ API pública

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        effort = request.reasoning_effort
        if effort is not None and effort not in _EFFORTS:
            raise LlmRequestError(f"reasoning_effort {effort!r} must be one of {sorted(_EFFORTS)}")
        output_schema = None
        if request.response_format:
            if request.response_format.get("type") != "json_schema":
                raise LlmUnsupportedError("provider='claude-code' only supports json_schema")
            output_schema = (request.response_format.get("json_schema") or {}).get("schema")
        registry = proto.tool_registry(request.tools)
        descriptions = {
            (t.get("function") or {}).get("name", ""): (t.get("function") or {}).get("description", "")
            for t in request.tools or []
        }
        schema, mode = proto.build_output_schema(registry, output_schema)
        ignored = [
            name
            for name, value in (("temperature", request.temperature), ("max_tokens", request.max_tokens))
            if value is not None
        ]
        note: str | None = None
        repairs = 0
        while True:
            stdin_text = proto.build_stdin(request.messages, registry, descriptions, repair_note=note)
            result = await self._run_cli(stdin_text, schema, effort)
            self._raise_for_result(result)
            try:
                decision = proto.parse_decision(
                    result["structured_output"],
                    registry,
                    output_schema,
                    max_calls=self.max_tool_calls,
                )
            except LlmProtocolError as error:
                # Reparar es seguro: aún no se ha devuelto ninguna llamada al Runner.
                if repairs >= self.protocol_repairs:
                    raise
                repairs += 1
                note = str(error)
                continue
            return LlmResponse(
                content=decision.text,
                tool_calls=decision.calls,
                finish_reason="tool_calls" if decision.calls else "stop",
                usage=self._usage(result),
                metrics=self._metrics(result, mode, repairs, ignored),
            )

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        """Decisión estructurada (no probabilística): sin confidence ni probabilities."""
        messages = build_structured_decision_messages(state, questions)
        schema = structured_decision_schema(questions)
        stdin_text = "\n\n".join(m["content"] for m in messages)
        result = await self._run_cli(stdin_text, schema, None)
        self._raise_for_result(result)
        answers = parse_decision_answers(json.dumps(result["structured_output"]), questions)
        return {"answers": answers, "usage": normalize_decision_usage(self._usage(result))}


def isolation_arg_limit() -> int:
    return proto.MAX_SCHEMA_BYTES


def parse_models(raw: Any) -> list[ModelInfo]:
    """Convierte ``models`` de la respuesta a ``initialize`` en ``ModelInfo`` (sin duplicados)."""
    models: list[ModelInfo] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("value"), str):
            continue
        model_id = item["value"]
        if model_id in seen:
            continue
        seen.add(model_id)
        efforts = item.get("supportedEffortLevels")
        label = str(item.get("displayName") or model_id)
        description = str(item["description"]) if item.get("description") else None
        # «Opus 5.5 · Best for…»: el nombre con versión es más útil que «Opus» a secas.
        if description and " · " in description and model_id != "default":
            versioned, _, rest = description.partition(" · ")
            label, description = versioned.strip() or label, rest.strip() or None
        models.append(
            ModelInfo(
                id=model_id,
                label=label,
                description=description,
                efforts=[str(e) for e in efforts] if isinstance(efforts, list) else [],
            )
        )
    return models
