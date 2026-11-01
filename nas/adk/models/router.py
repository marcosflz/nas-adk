from __future__ import annotations

import os
from typing import Any, AsyncGenerator, ClassVar, Literal

import httpx
from pydantic import Field, PrivateAttr, field_validator, model_validator

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse
from nas.adk.models.decision import DecisionOptions
from nas.adk.models.openai_compat import (
    DEFAULT_USER_AGENT,
    build_chat_body,
    build_responses_body,
    env_float,
    env_int,
    parse_models,
)
from nas.adk.models.opencode import OpenCodeLlm
from nas.adk.models.openrouter import OpenRouterLlm
from nas.adk.models.typesafe import TypesafeLlm

_DEFAULT_BASE_URLS = {
    "opencode": "https://opencode.ai/inference/go",
    "openrouter": "https://openrouter.ai/api/v1",
    "typesafe": "https://api.typesafe.ai/v1/systemone",
}
_API_KEYS = {
    "opencode": "OPENCODE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
}
_MODELS = {
    "opencode": "OPENCODE_MODEL",
    "openrouter": "OPENROUTER_MODEL",
    "typesafe": "JEV_MODEL",
}
_DEFAULT_MODELS = {
    "opencode": "deepseek-v4.1-flash",
    "openrouter": "",
    "typesafe": "jev-latest",
}
_BASE_URL_ENV = {
    "opencode": "OPENCODE_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
    "typesafe": "JEV_BASE_URL",
}
_API_ENV = {
    "opencode": "OPENCODE_API",
    "openrouter": "OPENROUTER_API",
    "typesafe": "",
}
_FALLBACK_ENV = {
    "opencode": "OPENCODE_FALLBACK_MODELS",
    "openrouter": "OPENROUTER_FALLBACK_MODELS",
    "typesafe": "",
}
_TIMEOUT_ENV = {
    "opencode": "OPENCODE_TIMEOUT",
    "openrouter": "OPENROUTER_TIMEOUT",
    "typesafe": "JEV_TIMEOUT",
}
_RETRIES_ENV = {
    "opencode": "OPENCODE_MAX_RETRIES",
    "openrouter": "OPENROUTER_MAX_RETRIES",
    "typesafe": "JEV_MAX_RETRIES",
}
_BACKOFF_ENV = {
    "opencode": "OPENCODE_BACKOFF",
    "openrouter": "OPENROUTER_BACKOFF",
    "typesafe": "JEV_BACKOFF",
}


class ModelRouter(BaseLlm):
    """Fachada async para OpenCode Go, OpenRouter y TypeSafe Jev.

    Foundry fue eliminado. ``provider="jev"`` se normaliza a ``"typesafe"``.
    OpenCode solo genera texto; las decisiones requieren OpenRouter o TypeSafe.
    """

    provider: Literal["opencode", "openrouter", "typesafe"] = "opencode"
    model: str = ""
    api_key: str = Field(default="", exclude=True, repr=False)
    base_url: str | None = None
    api: Literal["auto", "chat_completions", "responses"] = "auto"
    fallback_models: tuple[str, ...] = ()
    session_id: str | None = None
    user_agent: str = ""
    http_referer: str | None = None
    app_title: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    decision: DecisionOptions | None = None
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    backoff_base: float = Field(default=0.5, ge=0)
    http_client: httpx.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )

    supports_streaming: ClassVar[bool] = True
    _backend: BaseLlm | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider_and_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        provider = data.get("provider") or os.getenv("NAS_MODEL_PROVIDER", "opencode")
        if provider == "opencode-go":
            provider = "opencode"
        elif provider == "jev":
            provider = "typesafe"
        elif provider == "foundry":
            raise ValueError(
                "Foundry support was removed; use opencode, openrouter or typesafe"
            )
        if provider not in _DEFAULT_BASE_URLS:
            raise ValueError(
                f"Unsupported provider {provider!r}; use opencode, openrouter or typesafe"
            )
        data["provider"] = provider

        if not data.get("model"):
            data["model"] = os.getenv(_MODELS[provider], _DEFAULT_MODELS[provider])
        if not data.get("base_url"):
            provider_base = os.getenv(_BASE_URL_ENV[provider], "")
            data["base_url"] = (
                os.getenv("NAS_MODEL_BASE_URL", "")
                or provider_base
                or _DEFAULT_BASE_URLS[provider]
            )
        if not data.get("api_key"):
            data["api_key"] = os.getenv(_API_KEYS[provider], "")
        if not data.get("api"):
            data["api"] = os.getenv(
                _API_ENV[provider],
                "chat_completions" if provider == "openrouter" else "auto",
            )
        if data.get("fallback_models") is None:
            data["fallback_models"] = parse_models(os.getenv(_FALLBACK_ENV[provider]))
        if data.get("timeout") is None:
            data["timeout"] = env_float(_TIMEOUT_ENV[provider], 120.0)
        if data.get("max_retries") is None:
            data["max_retries"] = env_int(_RETRIES_ENV[provider], 2)
        if data.get("backoff_base") is None:
            data["backoff_base"] = env_float(_BACKOFF_ENV[provider], 0.5)
        if data.get("session_id") is None:
            data["session_id"] = os.getenv("OPENCODE_SESSION_ID") or None
        if data.get("http_referer") is None:
            data["http_referer"] = os.getenv("OPENROUTER_HTTP_REFERER") or None
        if data.get("app_title") is None:
            data["app_title"] = os.getenv("OPENROUTER_APP_TITLE") or None
        return data

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _normalize_fallbacks(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(model.strip() for model in value.split(",") if model.strip())
        return tuple(str(model).strip() for model in value if str(model).strip())

    @model_validator(mode="after")
    def _validate_configuration(self) -> ModelRouter:
        if not self.model:
            raise ValueError("OpenRouter requires an explicit model")
        if self.provider != "openrouter" and self.decision is not None:
            raise ValueError("decision options are only valid for provider='openrouter'")
        if self.provider == "typesafe":
            if self.fallback_models:
                raise ValueError("TypeSafe does not support fallback_models")
            if self.api != "auto":
                raise ValueError("TypeSafe does not support the api option")
        return self

    def _create_backend(self) -> BaseLlm:
        base_url = self.base_url or _DEFAULT_BASE_URLS[self.provider]
        user_agent = self.user_agent or DEFAULT_USER_AGENT
        if self.provider == "opencode":
            return OpenCodeLlm(
                model=self.model,
                api_key=self.api_key,
                base_url=base_url,
                http_client=self.http_client,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
                api=self.api,
                fallback_models=self.fallback_models,
                session_id=self.session_id,
                user_agent=user_agent,
                extra_body=self.extra_body,
                extra_headers=self.extra_headers,
            )
        if self.provider == "openrouter":
            return OpenRouterLlm(
                model=self.model,
                api_key=self.api_key,
                base_url=base_url,
                http_client=self.http_client,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
                api=self.api,
                fallback_models=self.fallback_models,
                user_agent=user_agent,
                http_referer=self.http_referer,
                app_title=self.app_title,
                extra_body=self.extra_body,
                extra_headers=self.extra_headers,
                decision=self.decision,
            )
        return TypesafeLlm(
            model=self.model,
            api_key=self.api_key,
            base_url=base_url,
            http_client=self.http_client,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            user_agent=user_agent,
        )

    def _get_backend(self) -> BaseLlm:
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        return await self._get_backend().generate_content_async(request)

    async def generate_stream_async(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmResponse, None]:
        async for chunk in self._get_backend().generate_stream_async(request):
            yield chunk

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._get_backend().decide_async(state, questions)

    async def aclose(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()
            self._backend = None


# Alias internos que conservan compatibilidad con tests y utilidades existentes.
_build_request_body = build_chat_body
_build_responses_body = build_responses_body

__all__ = ["ModelRouter"]
