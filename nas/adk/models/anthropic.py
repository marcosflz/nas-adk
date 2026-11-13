"""Proveedor ``anthropic``: API Messages oficial (facturación por API, tool calling nativo).

El proveedor traduce mensajes y herramientas, y devuelve llamadas normalizadas. No ejecuta
herramientas ni reintenta por su cuenta: los reintentos (con ``Retry-After``) son los del SDK.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import AsyncGenerator, Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field, PrivateAttr

from nas.adk.models.base_llm import (
    BaseLlm,
    LlmRequest,
    LlmResponse,
    ModelCapabilities,
    ToolCall,
)
from nas.adk.models.decision import (
    build_structured_decision_messages,
    normalize_decision_usage,
    parse_decision_answers,
    structured_decision_schema,
)
from nas.adk.models.errors import (
    LlmAuthError,
    LlmConnectionError,
    LlmError,
    LlmModelError,
    LlmPermissionError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmRefusalError,
    LlmRequestError,
    LlmResponseError,
    LlmServerError,
    LlmTimeoutError,
    LlmTruncatedError,
    LlmUnsupportedError,
)
from nas.adk.models.openai_compat import env_float, env_int

logger = logging.getLogger(__name__)

ITEM_MARKER = "nas_anthropic_message"
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
}
_STOP_MAP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls"}


def _import_sdk() -> Any:
    try:
        import anthropic
    except ImportError as error:  # pragma: no cover - depende del entorno
        raise LlmUnsupportedError(
            "provider='anthropic' requires the optional SDK: pip install 'nas-adk[anthropic]'"
        ) from error
    return anthropic


# --------------------------------------------------------------------------- mensajes


def _text_of(content: Any, role: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                kind = block.get("type") if isinstance(block, Mapping) else type(block).__name__
                raise LlmUnsupportedError(
                    f"provider='anthropic' only supports text content in {role} messages "
                    f"(got {kind!r}); non-text modalities are not implemented"
                )
        return "".join(parts)
    raise LlmRequestError(f"Unsupported {role} message content type: {type(content).__name__}")


def _is_error_result(text: str) -> bool:
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(data, dict) and set(data) == {"error"}


def _tool_result_content(text: str, images: Any) -> str | list[dict[str, Any]]:
    """Un resultado con ``images`` ([{media_type, data}]) lleva bloques de imagen en base64."""
    if not images:
        return text
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for image in images:
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["media_type"],
                    "data": image["data"],
                },
            }
        )
    return blocks


def to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convierte el historial OpenAI-like de NAS al formato Messages. Función pura."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending: list[str] = []  # tool_use ids sin resultado del último assistant
    results: list[dict[str, Any]] = []

    def flush_results() -> None:
        nonlocal results, pending
        if results:
            if set(r["tool_use_id"] for r in results) != set(pending) and pending:
                missing = sorted(set(pending) - {r["tool_use_id"] for r in results})
                raise LlmRequestError(f"Missing tool results for tool calls: {missing}")
            out.append({"role": "user", "content": results})
            results = []
            pending = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            text = _text_of(message.get("content"), "system")
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id or call_id not in pending:
                raise LlmRequestError(
                    f"Tool result {call_id!r} has no matching tool call in the previous assistant turn"
                )
            text = _text_of(message.get("content"), "tool")
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": _tool_result_content(text, message.get("images")),
            }
            if _is_error_result(text):
                block["is_error"] = True
            results.append(block)
            continue
        flush_results()
        if pending:
            raise LlmRequestError(f"Missing tool results for tool calls: {sorted(pending)}")
        if role == "user":
            text = _text_of(message.get("content"), "user")
            out.append({"role": "user", "content": [{"type": "text", "text": text}]})
        elif role == "assistant":
            blocks = _assistant_blocks(message)
            if not blocks:
                continue
            out.append({"role": "assistant", "content": blocks})
            pending = [b["id"] for b in blocks if b["type"] == "tool_use"]
        else:
            raise LlmRequestError(f"Unsupported message role: {role!r}")
    flush_results()
    if pending:
        raise LlmRequestError(f"Missing tool results for tool calls: {sorted(pending)}")
    if not out or out[0]["role"] != "user":
        raise LlmRequestError("The conversation must start with a user message")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    for item in message.get("_response_items") or []:
        if isinstance(item, Mapping) and item.get("type") == ITEM_MARKER:
            blocks = copy.deepcopy(item.get("content") or [])
            if blocks:
                return list(blocks)
    blocks = []
    text = _text_of(message.get("content"), "assistant")
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as error:
            raise LlmRequestError("Tool call arguments are not valid JSON") from error
        if not isinstance(args, dict):
            raise LlmRequestError("Tool call arguments must be a JSON object")
        name = function.get("name", "")
        if not _TOOL_NAME_RE.match(str(name)):
            raise LlmRequestError(f"Tool name {name!r} is not valid for provider='anthropic'")
        blocks.append({"type": "tool_use", "id": call.get("id", ""), "name": name, "input": args})
    return blocks


def to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        function = tool.get("function") or {}
        name = str(function.get("name", ""))
        if not _TOOL_NAME_RE.match(name):
            raise LlmRequestError(f"Tool name {name!r} is not valid for provider='anthropic'")
        converted.append(
            {
                "name": name,
                "description": function.get("description", "") or name,
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def sanitize_output_schema(schema: Any) -> Any:
    """Quita restricciones no admitidas por la salida estructurada (NAS las valida en local)."""
    if isinstance(schema, list):
        return [sanitize_output_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "minItems" and value not in (0, 1):
            continue
        if key == "default" or key == "title":
            continue
        cleaned[key] = sanitize_output_schema(value)
    if cleaned.get("type") == "object" or "properties" in cleaned:
        extra = cleaned.get("additionalProperties", False)
        if extra not in (False, None):
            raise LlmUnsupportedError(
                "Structured outputs require additionalProperties=false; "
                "this schema allows additional properties"
            )
        cleaned["additionalProperties"] = False
    return cleaned


def _output_config(request: LlmRequest) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if request.reasoning_effort:
        if request.reasoning_effort not in _EFFORTS:
            raise LlmRequestError(
                f"reasoning_effort {request.reasoning_effort!r} must be one of {sorted(_EFFORTS)}"
            )
        config["effort"] = request.reasoning_effort
    fmt = request.response_format
    if fmt:
        if fmt.get("type") != "json_schema":
            raise LlmUnsupportedError("provider='anthropic' only supports json_schema response_format")
        schema = (fmt.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise LlmRequestError("response_format.json_schema.schema must be an object")
        config["format"] = {"type": "json_schema", "schema": sanitize_output_schema(schema)}
    return config


# En estos modelos el razonamiento ya está activo por defecto, pero su texto llega vacío
# salvo que se pida resumido. En los anteriores activarlo cambiaría coste y comportamiento.
_THINKS_BY_DEFAULT = re.compile(r"^claude-[a-z]+-[5-9](?:[-.]|$)")


def summarized_thinking(model: str, request: LlmRequest) -> dict[str, Any] | None:
    if not request.reasoning_summary or not _THINKS_BY_DEFAULT.match(model):
        return None
    return {"type": "adaptive", "display": "summarized"}


# ------------------------------------------------------------------------ respuesta


def _usage(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    inp = getattr(usage, "input_tokens", None)
    outp = getattr(usage, "output_tokens", None)
    read = getattr(usage, "cache_read_input_tokens", None)
    created = getattr(usage, "cache_creation_input_tokens", None)
    if isinstance(inp, int):
        total_in = inp + (read or 0) + (created or 0)
        out["prompt_tokens"] = total_in
        out["input_tokens"] = total_in
    if isinstance(outp, int):
        out["completion_tokens"] = outp
        out["output_tokens"] = outp
    if "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    if isinstance(read, int):
        out["cache_read_input_tokens"] = read
    if isinstance(created, int):
        out["cache_creation_input_tokens"] = created
    return out


def parse_message(message: Any, *, ignored: list[str], model: str) -> LlmResponse:
    stop_reason = getattr(message, "stop_reason", None)
    blocks = [b.model_dump(mode="json", exclude_none=True) for b in message.content]
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    request_id = getattr(message, "_request_id", None)
    if stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise LlmRefusalError(
            "The model refused the request",
            category=getattr(details, "category", None),
        )
    if stop_reason == "max_tokens":
        raise LlmTruncatedError(
            "The response was truncated by max_tokens; raise max_tokens", partial_content=text
        )
    if stop_reason not in _STOP_MAP:
        raise LlmResponseError(f"Unsupported stop_reason from Anthropic: {stop_reason!r}")
    calls = [
        ToolCall(
            id=b["id"],
            function={"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
        )
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    finish = "tool_calls" if calls else "stop"
    return LlmResponse(
        content=text or None,
        tool_calls=calls,
        finish_reason=finish,
        usage=_usage(message),
        metrics={
            "provider": "anthropic",
            "model": getattr(message, "model", model),
            "request_id": request_id,
            "stop_reason": stop_reason,
            "ignored_parameters": ignored,
        },
        response_items=[{"type": ITEM_MARKER, "content": blocks}],
    )


# --------------------------------------------------------------------------- proveedor


class AnthropicLlm(BaseLlm):
    """API oficial de Anthropic mediante el SDK async."""

    provider: Literal["anthropic"] = "anthropic"
    model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", ""))
    api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""), repr=False, exclude=True
    )
    base_url: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", ""))
    timeout: float = Field(default_factory=lambda: env_float("ANTHROPIC_TIMEOUT", 120.0), gt=0)
    max_retries: int = Field(default_factory=lambda: env_int("ANTHROPIC_MAX_RETRIES", 2), ge=0)
    max_tokens_default: int = Field(
        default_factory=lambda: env_int("ANTHROPIC_MAX_TOKENS", 16000), gt=0
    )
    client: Any = None

    supports_streaming: ClassVar[bool] = True
    capabilities: ClassVar[ModelCapabilities | None] = ModelCapabilities(
        tool_calling="native",
        structured_output="native",
        decisions="structured",
        text_streaming=True,
        tool_argument_streaming=False,
        usage="reported",
        input_modalities=("text", "image"),
    )

    _client: Any = PrivateAttr(default=None)
    _owned: bool = PrivateAttr(default=False)

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._client is None:
            sdk = _import_sdk()
            if not self.api_key:
                raise LlmAuthError("ANTHROPIC_API_KEY is not set")
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = sdk.AsyncAnthropic(**kwargs)
            self._owned = True
        return self._client

    async def aclose(self) -> None:
        if self._owned and self._client is not None:
            await self._client.close()
        self._client = None
        self._owned = False

    def _params(self, request: LlmRequest) -> tuple[dict[str, Any], list[str]]:
        model = request.model or self.model
        if not model:
            raise LlmRequestError("provider='anthropic' requires an explicit model")
        system, messages = to_anthropic_messages(request.messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens or self.max_tokens_default,
            "messages": messages,
        }
        if system:
            params["system"] = system
        tools = to_anthropic_tools(request.tools)
        if tools:
            params["tools"] = tools
        output = _output_config(request)
        if output:
            params["output_config"] = output
        thinking = summarized_thinking(model, request)
        if thinking:
            params["thinking"] = thinking
        ignored: list[str] = []
        if request.temperature is not None:
            ignored.append("temperature")
            logger.warning("provider='anthropic' ignores temperature (rejected by the API)")
        return params, ignored

    def _wrap(self, error: BaseException) -> LlmError:
        if isinstance(error, LlmError):
            return error
        sdk = _import_sdk()
        if isinstance(error, sdk.APITimeoutError):
            return LlmTimeoutError("Anthropic request timed out", cause=error)
        if isinstance(error, sdk.APIConnectionError):
            return LlmConnectionError(f"Anthropic connection failed: {error}", cause=error)
        if isinstance(error, sdk.APIStatusError):
            status = error.status_code
            detail = getattr(error, "message", str(error))
            body = str(getattr(error, "body", ""))[:500]
            kwargs: dict[str, Any] = {"status_code": status, "body": body, "cause": error}
            if status == 401:
                return LlmAuthError(f"Anthropic authentication failed: {detail}", **kwargs)
            if status == 402:
                return LlmQuotaError(f"Anthropic billing problem: {detail}", **kwargs)
            if status == 403:
                return LlmPermissionError(f"Anthropic permission denied: {detail}", **kwargs)
            if status == 404:
                return LlmModelError(f"Anthropic model or endpoint not found: {detail}", **kwargs)
            if status in (400, 413, 422):
                return LlmRequestError(f"Anthropic rejected the request: {detail}", **kwargs)
            if status == 429:
                err = LlmRateLimitError(f"Anthropic rate limit: {detail}", **kwargs)
                err.retry_after = error.response.headers.get("retry-after")  # type: ignore[attr-defined]
                return err
            if status >= 500:
                return LlmServerError(f"Anthropic server error {status}: {detail}", **kwargs)
            return LlmResponseError(f"Anthropic error {status}: {detail}", **kwargs)
        return LlmResponseError(f"Unexpected Anthropic error: {error}", cause=error)

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        params, ignored = self._params(request)
        client = self._get_client()
        sdk = _import_sdk()
        try:
            message = await client.messages.create(**params)
            return parse_message(message, ignored=ignored, model=params["model"])
        except (sdk.APIError, sdk.AnthropicError) as error:
            raise self._wrap(error) from error

    async def generate_stream_async(self, request: LlmRequest) -> AsyncGenerator[LlmResponse, None]:
        params, ignored = self._params(request)
        client = self._get_client()
        sdk = _import_sdk()
        try:
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "text" and event.text:
                        yield LlmResponse(content=event.text, type="delta")
                    elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
                        if event.delta.thinking:
                            yield LlmResponse(reasoning=event.delta.thinking, type="delta")
                message = await stream.get_final_message()
            yield parse_message(message, ignored=ignored, model=params["model"])
        except (sdk.APIError, sdk.AnthropicError) as error:
            raise self._wrap(error) from error

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        """Decisión estructurada (no probabilística): sin confidence ni probabilities."""
        request = LlmRequest(
            model=self.model,
            messages=build_structured_decision_messages(state, questions),
            max_tokens=2000,
        )
        params, _ = self._params(request)
        params["output_config"] = {
            **params.get("output_config", {}),
            "format": {"type": "json_schema", "schema": structured_decision_schema(questions)},
        }
        client = self._get_client()
        sdk = _import_sdk()
        try:
            message = await client.messages.create(**params)
        except (sdk.APIError, sdk.AnthropicError) as error:
            raise self._wrap(error) from error
        response = parse_message(message, ignored=[], model=params["model"])
        return {
            "answers": parse_decision_answers(response.content, questions),
            "usage": normalize_decision_usage(response.usage),
        }
