from __future__ import annotations

import os
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.decision import (
    DecisionOptions,
    build_decision_messages,
    decision_schema,
    normalize_decision_usage,
    parse_decision_answers,
)
from nas.adk.models.errors import LlmError, LlmResponseError
from nas.adk.models.openai_compat import (
    DEFAULT_USER_AGENT,
    OpenAICompatibleLlm,
    env_float,
    env_int,
    parse_models,
)


class OpenRouterLlm(OpenAICompatibleLlm):
    """Proveedor OpenRouter para generación y decisiones estructuradas."""

    provider: Literal["openrouter"] = "openrouter"
    model: str = Field(min_length=1)
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
        exclude=True,
        repr=False,
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    api: Literal["auto", "chat_completions", "responses"] = Field(
        default_factory=lambda: _env_api()
    )
    fallback_models: tuple[str, ...] = Field(
        default_factory=lambda: parse_models(os.getenv("OPENROUTER_FALLBACK_MODELS"))
    )
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = Field(
        default_factory=lambda: env_float("OPENROUTER_TIMEOUT", 120.0), gt=0
    )
    max_retries: int = Field(
        default_factory=lambda: env_int("OPENROUTER_MAX_RETRIES", 2), ge=0
    )
    backoff_base: float = Field(
        default_factory=lambda: env_float("OPENROUTER_BACKOFF", 0.5), ge=0
    )
    http_referer: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_HTTP_REFERER") or None
    )
    app_title: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_APP_TITLE") or None
    )
    decision: DecisionOptions | None = None

    supports_streaming: ClassVar[bool] = True
    chat_path: ClassVar[str] = "/chat/completions"
    responses_path: ClassVar[str] = "/responses"

    @field_validator("api", mode="before")
    @classmethod
    def _validate_api(cls, value: Any) -> Any:
        if value in (None, ""):
            return "chat_completions"
        if value not in {"auto", "chat_completions", "responses"}:
            raise ValueError("OPENROUTER_API must be auto, chat_completions or responses")
        return value

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        headers = super()._headers(session_id)
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    def _decision_body(
        self,
        request: LlmRequest,
        model: str,
        questions: dict[str, Any],
        response_format: str,
    ) -> dict[str, Any]:
        options = self.decision or DecisionOptions()
        body = self._payload(request, model, "chat_completions")
        if response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "strict": options.strict,
                    "schema": decision_schema(questions),
                },
            }
        else:
            body["response_format"] = {"type": "json_object"}
        if options.require_parameters:
            provider = dict(body.get("provider") or {})
            provider["require_parameters"] = True
            body["provider"] = provider
        if options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens
        body["temperature"] = options.temperature
        return body

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        if not questions:
            raise LlmResponseError("OpenRouter requires at least one decision question")
        options = self.decision or DecisionOptions()
        request = LlmRequest(
            model=self.model,
            messages=build_decision_messages(state, questions),
        )
        candidates = self._candidate_models(request)
        last_error: LlmError | None = None
        response_format = options.response_format
        for candidate in candidates:
            for attempt in range(options.max_retries + 1):
                try:
                    body = self._decision_body(
                        request, candidate, questions, response_format
                    )
                    if response_format == "json_object":
                        body["messages"][0]["content"] += (
                            "\nReturn the JSON object now, without Markdown fences."
                        )
                    data = await self._post_json(
                        self._url("chat_completions"),
                        body,
                        self._headers(request.session_id or self.session_id),
                    )
                    if data.get("error"):
                        raise LlmResponseError(
                            str(data["error"]), retryable=True, fallback_allowed=True
                        )
                    parsed = self._parse(data, "chat_completions")
                    answers = parse_decision_answers(parsed.content, questions)
                    return {
                        "answers": answers,
                        "usage": normalize_decision_usage(parsed.usage),
                    }
                except LlmError as error:
                    last_error = error
                    if (
                        error.status_code == 400
                        and response_format == "json_schema"
                    ):
                        # Algunos modelos/proveedores no aceptan json_schema.
                        response_format = "json_object"
                        continue
                    if isinstance(error, LlmResponseError) and error.status_code is None:
                        error.retryable = True
                        error.fallback_allowed = True
                    if error.retryable and attempt < options.max_retries:
                        continue
                    if error.fallback_allowed:
                        break
                    raise
        if last_error is not None:
            raise last_error
        raise LlmResponseError("OpenRouter produced no decision")


def _env_api() -> Literal["auto", "chat_completions", "responses"]:
    value = os.getenv("OPENROUTER_API", "chat_completions")
    if value not in {"auto", "chat_completions", "responses"}:
        return "chat_completions"
    return value  # type: ignore[return-value]


__all__ = ["OpenRouterLlm"]
