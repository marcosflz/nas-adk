"""Proveedor ``chatgpt``: el plan de ChatGPT de una persona vía «Sign in with ChatGPT».

Inferencia por HTTP en ``POST https://api.openai.com/v1/responses`` con el ``access_token``
que OpenAI emite a esa persona (véase :mod:`nas.adk.models.chatgpt.oauth`). Las condiciones de
la ruta: ``store: false`` y ``stream: true`` en cada petición, instrucciones en ``instructions``
(no se aceptan mensajes ``system``), herramientas de función agrupadas en un *namespace* y sin
``temperature``, ``max_output_tokens``, ``previous_response_id`` ni similares.

Cada token es de una persona: úsalo solo para sus propias peticiones.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, ClassVar, Literal

import httpx
from pydantic import Field

from nas.adk.models.base_llm import LlmRequest, LlmResponse, ModelCapabilities
from nas.adk.models.errors import (
    LlmAuthError,
    LlmError,
    LlmPermissionError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmRequestError,
    LlmResponseError,
)
from nas.adk.models.openai_compat import OpenAICompatibleLlm, build_responses_body, error_from_status

API_BASE = "https://api.openai.com/v1"
TOOL_NAMESPACE = "nexus_tools"

# Campos que la ruta de planes de ChatGPT rechaza (documentados por OpenAI).
_REJECTED_FIELDS = (
    "background",
    "conversation",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "moderation",
    "multi_agent",
    "prompt",
    "prompt_cache_retention",
    "safety_identifier",
    "temperature",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
    "previous_response_id",
    "parallel_tool_calls",
)


def plan_body(body: dict[str, Any]) -> dict[str, Any]:
    """Adapta un cuerpo de Responses a las reglas de la ruta de planes de ChatGPT."""
    for name in _REJECTED_FIELDS:
        body.pop(name, None)
    instructions: list[str] = []
    items: list[Any] = []
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("role") in {"system", "developer"} and "type" not in item:
            content = item.get("content")
            if isinstance(content, str) and content:
                instructions.append(content)
            continue
        if isinstance(item, dict) and item.get("type") == "function_call" and "namespace" not in item:
            item = {**item, "namespace": TOOL_NAMESPACE}
        items.append(item)
    body["input"] = items
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    tools = body.pop("tools", None)
    if tools:
        body["tools"] = [
            {
                "type": "namespace",
                "name": TOOL_NAMESPACE,
                "description": "Herramientas de la aplicación.",
                "tools": [{key: value for key, value in tool.items() if key != "strict"} | {"strict": False} for tool in tools],
            }
        ]
    body["store"] = False
    body["stream"] = True
    return body


def plan_error(status: int, body: str | None, response: httpx.Response) -> LlmError:
    """Traduce los códigos documentados de la ruta de planes; si no hay, por estado HTTP."""
    code = ""
    param = None
    try:
        data = json.loads(body or "")
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            code = str(error.get("code") or "")
            param = error.get("param")
    except ValueError:
        pass
    message = f"ChatGPT plan request failed ({status}{', ' + code if code else ''})"
    if code == "subscription_sharing_usage_limit_exceeded":
        return LlmQuotaError(
            "Se ha alcanzado el límite de uso de tu plan de ChatGPT. Revisa ChatGPT → Ajustes → Uso.",
            status_code=status,
            retryable=False,
            fallback_allowed=False,
            body=body,
        )
    if code in {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"}:
        return LlmRateLimitError(message, status_code=status, retryable=True, fallback_allowed=False, body=body)
    if code == "subscription_sharing_invalid_user" or status == 401:
        return LlmAuthError(message, status_code=status, body=body)
    if code == "subscription_sharing_unsupported_capability":
        return LlmRequestError(f"{message}: unsupported {param or 'capability'}", status_code=status, body=body)
    if code.startswith(("subscription_sharing_", "chatpass_")) or status == 403:
        return LlmPermissionError(message, status_code=status, retryable=False, fallback_allowed=False, body=body)
    return error_from_status(response, provider="chatgpt", body=body)


class ChatGptPlanLlm(OpenAICompatibleLlm):
    """Plan de ChatGPT de una persona (Responses API, herramientas nativas, streaming)."""

    provider: str = "chatgpt"
    base_url: str = API_BASE
    api: Literal["auto", "chat_completions", "responses"] = "responses"
    access_token: str = Field(default="", exclude=True, repr=False)
    max_retries: int = Field(default=1, ge=0)

    supports_streaming: ClassVar[bool] = True
    capabilities: ClassVar[ModelCapabilities | None] = ModelCapabilities(
        tool_calling="native",
        structured_output="native",
        decisions="none",
        text_streaming=True,
        tool_argument_streaming=False,
        usage="reported",
        input_modalities=("text", "image"),
        personal_use_only=True,
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.access_token and not self.api_key:
            self.api_key = self.access_token

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        if not self.api_key:
            raise LlmAuthError("ChatGPT access token is not configured")
        headers = super()._headers(session_id)
        headers["Accept"] = "text/event-stream"
        return headers

    def _payload(
        self,
        request: LlmRequest,
        model: str,
        api: Literal["chat_completions", "responses"],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        return plan_body(build_responses_body(request, model=model) | {"model": model})

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        """La ruta solo admite streaming: se consume el stream y se devuelve la respuesta final."""
        final: LlmResponse | None = None
        async for chunk in self.generate_stream_async(request):
            if chunk.type == "response":
                final = chunk
        if final is None:
            raise LlmResponseError("ChatGPT returned no final response")
        return final

    async def generate_stream_async(self, request: LlmRequest) -> AsyncGenerator[LlmResponse, None]:
        try:
            async for chunk in super().generate_stream_async(request):
                yield chunk
        except LlmError as error:
            raise _translate(error) from error


def _translate(error: LlmError) -> LlmError:
    """Errores HTTP o eventos ``response.failed`` con código de la ruta de planes → error tipado."""
    if getattr(error, "_chatgpt_plan", False) or not error.body:
        return error
    status = error.status_code or 400
    translated = plan_error(status, error.body, httpx.Response(status))
    if type(translated) is LlmResponseError and error.status_code is None:
        return error
    translated._chatgpt_plan = True  # type: ignore[attr-defined]
    return translated


def parse_models(data: Mapping[str, Any]) -> list[dict[str, str]]:
    """``GET /v1/models`` de la ruta de planes: solo ``visibility == "list"``, en el orden dado."""
    raw = data.get("models")
    if raw is None:
        raw = data.get("data")
    models: list[dict[str, str]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug") or item.get("id")
        if not isinstance(slug, str) or ("visibility" in item and item.get("visibility") != "list"):
            continue
        models.append({"id": slug, "label": str(item.get("display_name") or slug)})
    return models
