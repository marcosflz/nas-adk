from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any, ClassVar, Literal

import httpx
from pydantic import Field, PrivateAttr, field_validator

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ToolCall
from nas.adk.models.errors import (
    LlmAuthError,
    LlmError,
    LlmModelError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmResponseError,
    LlmServerError,
)
from nas.adk.version import __version__

DEFAULT_USER_AGENT = f"nas-adk/{__version__}"


def parse_models(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(model.strip() for model in value.split(",") if model.strip())
    return tuple(str(model).strip() for model in value if str(model).strip())


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_chat_body(request: LlmRequest, *, model: str | None = None) -> dict[str, Any]:
    body = request.model_dump(
        include={
            "model",
            "messages",
            "tools",
            "max_tokens",
            "temperature",
            "response_format",
            "reasoning_effort",
        },
        exclude_none=True,
    )
    if model:
        body["model"] = model
    body["messages"] = _chat_messages(request.messages)
    return body


def _data_url(image: Mapping[str, Any]) -> str:
    return f"data:{image['media_type']};base64,{image['data']}"


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """El rol ``tool`` solo admite texto: las imágenes van en un mensaje de usuario justo detrás."""
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        if message.get("role") != "tool":
            flush()
        out.append({k: v for k, v in message.items() if k not in ("_response_items", "images")})
        for image in message.get("images") or []:
            pending.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
    flush()
    return out


def build_responses_body(request: LlmRequest, *, model: str | None = None) -> dict[str, Any]:
    body = build_chat_body(request, model=model)
    body.pop("messages", None)
    body["input"] = responses_input(request.messages)
    body["store"] = False
    body["include"] = ["reasoning.encrypted_content"]
    tools = body.pop("tools", None)
    if tools:
        if any(tool.get("type") != "function" for tool in tools):
            raise ValueError("Responses adapter only supports function tools")
        body["tools"] = [
            {"type": "function", "strict": False, **tool["function"]} for tool in tools
        ]
    if "reasoning_effort" in body:
        body["reasoning"] = {"effort": body.pop("reasoning_effort")}
    if "max_tokens" in body:
        body["max_output_tokens"] = body.pop("max_tokens")
    if "response_format" in body:
        fmt = body.pop("response_format")
        if fmt.get("type") == "json_schema":
            fmt = {"type": "json_schema", **fmt["json_schema"]}
        body["text"] = {"format": fmt}
    return body


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            items.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        role = message["role"]
        if role != "tool":
            flush()
        if role == "assistant" and message.get("_response_items"):
            items.extend(
                {key: value for key, value in item.items() if key != "status"}
                for item in message["_response_items"]
            )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content") or "",
                }
            )
            for image in message.get("images") or []:
                pending.append({"type": "input_image", "image_url": _data_url(image)})
        elif role in {"user", "assistant", "system", "developer"}:
            if message.get("content") or role != "assistant":
                items.append({"role": role, "content": message.get("content") or ""})
            for call in message.get("tool_calls") or []:
                function = call["function"]
                args = function.get("arguments", "{}")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": (
                            args
                            if isinstance(args, str)
                            else json.dumps(args, ensure_ascii=False)
                        ),
                    }
                )
        else:
            raise ValueError(f"Unsupported Responses message role: {role}")
    flush()
    return items


def parse_chat_response(data: Mapping[str, Any]) -> LlmResponse:
    choices = data.get("choices")
    if not choices or not isinstance(choices, list) or "message" not in choices[0]:
        raise LlmResponseError("Invalid Chat Completions response: missing choices/message")
    choice = choices[0]
    message = choice["message"]
    return LlmResponse(
        content=message.get("content"),
        tool_calls=[
            ToolCall(id=call.get("id", ""), function=call["function"])
            for call in message.get("tool_calls") or []
        ],
        finish_reason=choice.get("finish_reason") or "stop",
        usage=dict(data.get("usage") or {}),
    )


async def parse_chat_stream(
    chunks: AsyncIterator[dict[str, Any]],
) -> AsyncGenerator[LlmResponse, None]:
    content = ""
    calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    usage: dict[str, Any] = {}
    async for data in chunks:
        if data.get("error"):
            raise LlmResponseError("Chat Completions stream returned an error")
        usage.update(data.get("usage") or {})
        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        # Varios servidores compatibles publican el razonamiento en un campo aparte.
        thought = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(thought, str) and thought:
            yield LlmResponse(reasoning=thought, type="delta")
        text = delta.get("content") or ""
        if text:
            content += text
            yield LlmResponse(content=text, type="delta")
        for tool_call in delta.get("tool_calls") or []:
            call = calls.setdefault(
                tool_call.get("index", 0),
                {"id": "", "function": {"name": "", "arguments": ""}},
            )
            if tool_call.get("id"):
                call["id"] = tool_call["id"]
            function = tool_call.get("function") or {}
            for key in ("name", "arguments"):
                call["function"][key] += function.get(key) or ""
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    if finish is None:
        raise LlmResponseError("Incomplete Chat Completions stream")
    yield parse_chat_response(
        {
            "choices": [
                {
                    "message": {
                        "content": content or None,
                        "tool_calls": [calls[i] for i in sorted(calls)],
                    },
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        }
    )


def parse_responses_response(data: Mapping[str, Any]) -> LlmResponse:
    if data.get("error") or data.get("status") in {"failed", "cancelled"}:
        error = data.get("error") or {}
        raise LlmResponseError(
            str(error.get("message") or f"Responses request {data.get('status')}")
        )
    output = data.get("output")
    if not isinstance(output, list) or not all(isinstance(item, dict) for item in output):
        raise LlmResponseError("Invalid Responses response: missing output items")
    content: list[str] = []
    final_content: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if item.get("type") == "message":
            if item.get("role") != "assistant":
                raise LlmResponseError("Invalid Responses output message role")
            parts = item.get("content")
            if not isinstance(parts, list) or not all(
                isinstance(part, dict) for part in parts
            ):
                raise LlmResponseError("Invalid Responses output message content")
            text: list[str] = []
            for part in parts:
                if part.get("type") == "output_text":
                    text.append(part["text"])
                elif part.get("type") == "refusal":
                    text.append(part["refusal"])
                else:
                    raise LlmResponseError(
                        f"Unsupported Responses content type: {part.get('type')}"
                    )
            message_text = "".join(text)
            content.append(message_text)
            if item.get("phase") == "final_answer":
                final_content.append(message_text)
        elif item.get("type") == "function_call":
            if (
                not item.get("call_id")
                or not item.get("name")
                or not isinstance(item.get("arguments"), str)
            ):
                raise LlmResponseError("Invalid Responses function call")
            calls.append(
                ToolCall(
                    id=item["call_id"],
                    function={"name": item["name"], "arguments": item["arguments"]},
                )
            )
        elif item.get("type") != "reasoning":
            raise LlmResponseError(f"Unsupported Responses output type: {item.get('type')}")
    usage = data.get("usage") or {}
    normalized = {
        target: usage[original]
        for original, target in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        )
        if isinstance(usage.get(original), int)
    }
    for original, target in (
        ("input_tokens_details", "prompt_tokens_details"),
        ("output_tokens_details", "completion_tokens_details"),
    ):
        if isinstance(usage.get(original), dict):
            normalized[target] = usage[original]
    status = data.get("status")
    finish = (
        ("tool_calls" if calls else "stop")
        if status == "completed"
        else (status or "incomplete")
    )
    if status == "incomplete":
        finish = (data.get("incomplete_details") or {}).get("reason") or "incomplete"
    return LlmResponse(
        content="\n\n".join(final_content or content) or None,
        tool_calls=calls,
        finish_reason=finish,
        usage=normalized,
        response_items=output,
    )


async def parse_responses_stream(
    chunks: AsyncIterator[dict[str, Any]],
) -> AsyncGenerator[LlmResponse, None]:
    last_item: str | None = None
    async for event in chunks:
        kind = event.get("type")
        if kind in {"response.output_text.delta", "response.refusal.delta"}:
            if event.get("delta"):
                item = event.get("item_id")
                if isinstance(item, str):
                    if last_item is not None and item != last_item:
                        yield LlmResponse(content="\n\n", type="delta")
                    last_item = item
                yield LlmResponse(content=event["delta"], type="delta")
        elif kind == "response.reasoning_summary_text.delta":
            if event.get("delta"):
                yield LlmResponse(reasoning=event["delta"], type="delta")
        elif kind in {"response.completed", "response.incomplete", "response.failed"}:
            yield parse_responses_response(event["response"])
            return
        elif kind == "error":
            raise LlmResponseError(
                str(event.get("message") or "Responses stream returned an error")
            )
    raise LlmResponseError("Incomplete Responses stream: missing terminal event")


async def iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                data_lines.clear()
                if raw.strip() == "[DONE]":
                    return
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise LlmResponseError("The stream returned invalid JSON") from error
                if not isinstance(parsed, dict):
                    raise LlmResponseError("The stream returned a non-object JSON event")
                yield parsed
    if data_lines:
        raw = "\n".join(data_lines)
        if raw.strip() != "[DONE]":
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                raise LlmResponseError("The stream returned invalid JSON") from error
            if not isinstance(parsed, dict):
                raise LlmResponseError("The stream returned a non-object JSON event")
            yield parsed


def error_from_status(
    response: httpx.Response,
    *,
    provider: str,
    body: str | None = None,
) -> LlmError:
    status = response.status_code
    message = f"{provider} returned HTTP {status}"
    if status == 401:
        return LlmAuthError(message, status_code=status, body=body)
    if status == 402:
        return LlmQuotaError(
            message, status_code=status, retryable=False, fallback_allowed=True, body=body
        )
    if status in (403, 404):
        return LlmModelError(
            message, status_code=status, retryable=False, fallback_allowed=True, body=body
        )
    if status == 429:
        return LlmRateLimitError(
            message, status_code=status, retryable=True, fallback_allowed=True, body=body
        )
    if status >= 500 or status in (408, 409, 425):
        return LlmServerError(
            message, status_code=status, retryable=True, fallback_allowed=True, body=body
        )
    return LlmResponseError(message, status_code=status, body=body)


class OpenAICompatibleLlm(BaseLlm):
    """Base async para proveedores compatibles con Chat Completions/Responses."""

    provider: str = ""
    model: str = ""
    api_key: str = Field(default="", exclude=True, repr=False)
    base_url: str = ""
    api: Literal["auto", "chat_completions", "responses"] = "auto"
    fallback_models: tuple[str, ...] = ()
    session_id: str | None = None
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    backoff_base: float = Field(default=0.5, ge=0)
    http_client: httpx.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )
    extra_body: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    supports_streaming: ClassVar[bool] = True
    chat_path: ClassVar[str] = "/chat/completions"
    responses_path: ClassVar[str] = "/responses"

    _client: httpx.AsyncClient | None = PrivateAttr(default=None)

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _normalize_fallbacks(cls, value: Any) -> tuple[str, ...]:
        return parse_models(value)

    def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is not None:
            return self.http_client
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _api_for_model(self, model: str) -> Literal["chat_completions", "responses"]:
        if self.api != "auto":
            return self.api
        if model.lower().startswith(("gpt-", "grok-", "muse-")):
            return "responses"
        return "chat_completions"

    def _url(self, api: Literal["chat_completions", "responses"]) -> str:
        path = self.chat_path if api == "chat_completions" else self.responses_path
        return f"{self.base_url.rstrip('/')}{path}"

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        if not self.api_key:
            raise LlmAuthError(f"{self.provider} API key is not configured")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        request: LlmRequest,
        model: str,
        api: Literal["chat_completions", "responses"],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        if api == "responses":
            body = build_responses_body(request, model=model)
        else:
            body = build_chat_body(request, model=model)
        body.update(_jsonable(self.extra_body))
        body["model"] = model
        if stream:
            body["stream"] = True
            if api == "chat_completions":
                body["stream_options"] = {"include_usage": True}
        return body

    async def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> dict[str, Any]:
        try:
            response = await self._get_client().post(
                url, json=_jsonable(dict(payload)), headers=dict(headers), timeout=self.timeout
            )
        except httpx.TimeoutException as error:
            raise LlmServerError(
                f"{self.provider} timed out after {self.timeout:g}s",
                retryable=True,
                fallback_allowed=True,
                cause=error,
            ) from error
        except httpx.HTTPError as error:
            raise LlmServerError(
                f"{self.provider} network error: {type(error).__name__}",
                retryable=True,
                fallback_allowed=True,
                cause=error,
            ) from error
        if response.status_code >= 400:
            raise error_from_status(
                response,
                provider=self.provider,
                body=response.text[:500] if response.text else None,
            )
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise LlmResponseError(
                f"{self.provider} returned non-JSON content",
                status_code=response.status_code,
                retryable=True,
                fallback_allowed=True,
                cause=error,
            ) from error
        if not isinstance(data, dict):
            raise LlmResponseError(
                f"{self.provider} returned non-object JSON",
                status_code=response.status_code,
                retryable=True,
                fallback_allowed=True,
            )
        return data

    def _metrics(
        self,
        *,
        api: str,
        model: str,
        attempted_models: list[str],
        usage: Mapping[str, Any],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "provider": self.provider,
            "api": api,
            "model": model,
            "attempted_models": [*attempted_models, model],
            "fallback_used": bool(attempted_models),
        }
        if self.provider == "opencode" and isinstance(usage.get("opencode"), Mapping):
            opencode = usage["opencode"]
            metrics["billing"] = {
                "source": opencode.get("billing_source") or "go",
                "allowance_usd": opencode.get("allowance"),
                "list_cost_usd": opencode.get("list_cost"),
                "cost_usd": usage.get("cost"),
            }
        elif self.provider == "openrouter":
            metrics["billing"] = {
                "source": "openrouter",
                "cost_usd": usage.get("cost"),
            }
        return metrics

    def _candidate_models(self, request: LlmRequest) -> list[str]:
        primary = request.model or self.model
        ordered: list[str] = []
        for candidate in (primary, *self.fallback_models):
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        if not ordered:
            raise LlmResponseError(f"{self.provider} requires a model")
        return ordered

    def _parse(self, data: Mapping[str, Any], api: str) -> LlmResponse:
        try:
            response = (
                parse_responses_response(data)
                if api == "responses"
                else parse_chat_response(data)
            )
            raw_usage = data.get("usage")
            if isinstance(raw_usage, Mapping):
                for key in ("opencode", "cost"):
                    if key in raw_usage and key not in response.usage:
                        response.usage[key] = raw_usage[key]
            return response
        except LlmResponseError as error:
            if error.retryable or error.fallback_allowed:
                raise
            raise LlmResponseError(
                str(error), retryable=True, fallback_allowed=True, cause=error
            ) from error
        except (KeyError, TypeError, ValueError) as error:
            raise LlmResponseError(
                f"Invalid {self.provider} response: {error}",
                retryable=True,
                fallback_allowed=True,
                cause=error,
            ) from error

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        candidates = self._candidate_models(request)
        attempted: list[str] = []
        last_error: LlmError | None = None
        for index, candidate in enumerate(candidates):
            api = self._api_for_model(candidate)
            for attempt in range(self.max_retries + 1):
                try:
                    data = await self._post_json(
                        self._url(api),
                        self._payload(request, candidate, api),
                        self._headers(request.session_id or self.session_id),
                    )
                    response = self._parse(data, api)
                    response.metrics = self._metrics(
                        api=api,
                        model=candidate,
                        attempted_models=attempted,
                        usage=response.usage,
                    )
                    return response
                except LlmError as error:
                    last_error = error
                    if error.retryable and attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_base * (2**attempt))
                        continue
                    if error.fallback_allowed and index < len(candidates) - 1:
                        attempted.append(candidate)
                        break
                    raise
        if last_error is not None:
            raise last_error
        raise LlmResponseError(f"{self.provider} produced no response")

    async def generate_stream_async(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmResponse, None]:
        candidates = self._candidate_models(request)
        attempted: list[str] = []
        last_error: LlmError | None = None
        for index, candidate in enumerate(candidates):
            api = self._api_for_model(candidate)
            for attempt in range(self.max_retries + 1):
                emitted = False
                failure: LlmError
                try:
                    async with self._get_client().stream(
                        "POST",
                        self._url(api),
                        json=_jsonable(self._payload(request, candidate, api, stream=True)),
                        headers=self._headers(request.session_id or self.session_id),
                        timeout=self.timeout,
                    ) as response:
                        if response.status_code >= 400:
                            body = (await response.aread()).decode("utf-8", "replace")[:500]
                            raise error_from_status(
                                response, provider=self.provider, body=body
                            )
                        parser = (
                            parse_responses_stream(iter_sse(response))
                            if api == "responses"
                            else parse_chat_stream(iter_sse(response))
                        )
                        async for chunk in parser:
                            emitted = True
                            if chunk.type == "response":
                                chunk.metrics = self._metrics(
                                    api=api,
                                    model=candidate,
                                    attempted_models=attempted,
                                    usage=chunk.usage,
                                )
                            yield chunk
                    return
                except LlmError as error:
                    last_error = error
                    if emitted:
                        raise
                    if error.retryable and attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_base * (2**attempt))
                        continue
                    if error.fallback_allowed and index < len(candidates) - 1:
                        attempted.append(candidate)
                        break
                    raise
                except httpx.TimeoutException as error:
                    failure = LlmServerError(
                        f"{self.provider} timed out after {self.timeout:g}s",
                        retryable=True,
                        fallback_allowed=True,
                        cause=error,
                    )
                    last_error = failure
                    if emitted:
                        raise failure from error
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_base * (2**attempt))
                        continue
                    if index < len(candidates) - 1:
                        attempted.append(candidate)
                        break
                    raise failure from error
                except httpx.HTTPError as error:
                    failure = LlmServerError(
                        f"{self.provider} network error: {type(error).__name__}",
                        retryable=True,
                        fallback_allowed=True,
                        cause=error,
                    )
                    last_error = failure
                    if emitted:
                        raise failure from error
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_base * (2**attempt))
                        continue
                    if index < len(candidates) - 1:
                        attempted.append(candidate)
                        break
                    raise failure from error
                except (ValueError, KeyError, TypeError) as error:
                    failure = LlmResponseError(
                        f"Invalid {self.provider} stream: {error}",
                        retryable=True,
                        fallback_allowed=True,
                        cause=error,
                    )
                    last_error = failure
                    if emitted:
                        raise failure from error
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_base * (2**attempt))
                        continue
                    if index < len(candidates) - 1:
                        attempted.append(candidate)
                        break
                    raise failure from error
        if last_error is not None:
            raise last_error
        raise LlmResponseError(f"{self.provider} produced no stream")


__all__ = [
    "DEFAULT_USER_AGENT",
    "OpenAICompatibleLlm",
    "build_chat_body",
    "build_responses_body",
    "env_float",
    "env_int",
    "error_from_status",
    "iter_sse",
    "parse_chat_response",
    "parse_chat_stream",
    "parse_models",
    "parse_responses_response",
    "parse_responses_stream",
    "responses_input",
]
