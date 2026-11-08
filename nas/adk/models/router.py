from __future__ import annotations

import os
from typing import Any, AsyncGenerator, ClassVar, Literal

import httpx
from pydantic import Field, PrivateAttr, field_validator, model_validator

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ModelCapabilities
from nas.adk.models.decision import DecisionOptions
from nas.adk.models.openai_compat import (
    DEFAULT_USER_AGENT,
    build_chat_body,
    build_responses_body,
    env_float,
    env_int,
    parse_models,
)
from nas.adk.models.anthropic import AnthropicLlm
from nas.adk.models.claude_code import ClaudeCodeLlm
from nas.adk.models.generic import GenericOpenAILlm
from nas.adk.models.opencode import OpenCodeLlm
from nas.adk.models.openrouter import OpenRouterLlm
from nas.adk.models.typesafe import TypesafeLlm

_DEFAULT_BASE_URLS = {
    "opencode": "https://opencode.ai/inference/go",
    "openrouter": "https://openrouter.ai/api/v1",
    "typesafe": "https://api.typesafe.ai/v1/systemone",
    # Sin valor por defecto: hay que indicar base_url (OpenAI, Gemini, Ollama…).
    "openai-compatible": "",
    # El SDK usa su URL por defecto; claude-code no tiene URL (proceso local).
    "anthropic": "",
    "claude-code": "",
}
_API_KEYS = {
    "opencode": "OPENCODE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude-code": "",
}
_MODELS = {
    "opencode": "OPENCODE_MODEL",
    "openrouter": "OPENROUTER_MODEL",
    "typesafe": "JEV_MODEL",
    "openai-compatible": "OPENAI_COMPATIBLE_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "claude-code": "CLAUDE_CODE_MODEL",
}
_DEFAULT_MODELS = {
    "opencode": "deepseek-v4.1-flash",
    "openrouter": "",
    "typesafe": "jev-latest",
    "openai-compatible": "",
    "anthropic": "",
    "claude-code": "",
}
_BASE_URL_ENV = {
    "opencode": "OPENCODE_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
    "typesafe": "JEV_BASE_URL",
    "openai-compatible": "OPENAI_COMPATIBLE_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "claude-code": "",
}
_API_ENV = {
    "opencode": "OPENCODE_API",
    "openrouter": "OPENROUTER_API",
    "typesafe": "",
    "openai-compatible": "",
    "anthropic": "",
    "claude-code": "",
}
_FALLBACK_ENV = {
    "opencode": "OPENCODE_FALLBACK_MODELS",
    "openrouter": "OPENROUTER_FALLBACK_MODELS",
    "typesafe": "",
    "openai-compatible": "",
    "anthropic": "",
    "claude-code": "",
}
_TIMEOUT_ENV = {
    "opencode": "OPENCODE_TIMEOUT",
    "openrouter": "OPENROUTER_TIMEOUT",
    "typesafe": "JEV_TIMEOUT",
    "openai-compatible": "OPENAI_COMPATIBLE_TIMEOUT",
    "anthropic": "ANTHROPIC_TIMEOUT",
    "claude-code": "CLAUDE_CODE_TIMEOUT",
}
_RETRIES_ENV = {
    "opencode": "OPENCODE_MAX_RETRIES",
    "openrouter": "OPENROUTER_MAX_RETRIES",
    "typesafe": "JEV_MAX_RETRIES",
    "openai-compatible": "OPENAI_COMPATIBLE_MAX_RETRIES",
    "anthropic": "ANTHROPIC_MAX_RETRIES",
    "claude-code": "",
}
_BACKOFF_ENV = {
    "opencode": "OPENCODE_BACKOFF",
    "openrouter": "OPENROUTER_BACKOFF",
    "typesafe": "JEV_BACKOFF",
    "openai-compatible": "OPENAI_COMPATIBLE_BACKOFF",
    "anthropic": "",
    "claude-code": "",
}
# Proveedores sin URL configurable ni NAS_MODEL_BASE_URL (evita enviar credenciales a otro host).
_NO_SHARED_BASE_URL = {"anthropic", "claude-code"}
_OPTIONAL_MODEL = {"claude-code"}


class ModelRouter(BaseLlm):
    """Fachada async para OpenCode Go, OpenRouter, TypeSafe Jev y endpoints OpenAI-compatibles.

    Foundry fue eliminado. ``provider="jev"`` se normaliza a ``"typesafe"``.
    OpenCode solo genera texto; las decisiones requieren OpenRouter o TypeSafe.
    """

    provider: Literal[
        "opencode", "openrouter", "typesafe", "openai-compatible", "anthropic", "claude-code"
    ] = "opencode"
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
        elif provider == "claude":
            raise ValueError(
                "provider='claude' is ambiguous (API billing vs subscription); "
                "choose 'anthropic' (API) or 'claude-code' (local Claude Code subscription)"
            )
        if provider not in _DEFAULT_BASE_URLS:
            raise ValueError(
                f"Unsupported provider {provider!r}; use opencode, openrouter, typesafe, "
                "openai-compatible, anthropic or claude-code"
            )
        data["provider"] = provider

        if not data.get("model"):
            data["model"] = os.getenv(_MODELS[provider], _DEFAULT_MODELS[provider])
        if not data.get("base_url"):
            provider_base = os.getenv(_BASE_URL_ENV[provider], "")
            shared = "" if provider in _NO_SHARED_BASE_URL else os.getenv("NAS_MODEL_BASE_URL", "")
            data["base_url"] = shared or provider_base or _DEFAULT_BASE_URLS[provider]
        if not data.get("api_key") and _API_KEYS[provider]:
            data["api_key"] = os.getenv(_API_KEYS[provider], "")
        if not data.get("api"):
            data["api"] = os.getenv(
                _API_ENV[provider],
                "chat_completions" if provider == "openrouter" else "auto",
            )
        if data.get("fallback_models") is None:
            data["fallback_models"] = parse_models(os.getenv(_FALLBACK_ENV[provider]))
        if data.get("timeout") is None:
            data["timeout"] = env_float(
                _TIMEOUT_ENV[provider], 300.0 if provider == "claude-code" else 120.0
            )
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
        if not self.model and self.provider not in _OPTIONAL_MODEL:
            raise ValueError(f"{self.provider} requires an explicit model")
        if self.provider == "claude-code":
            for name, bad in (
                ("api_key", bool(self.api_key)),
                ("base_url", bool(self.base_url)),
                ("fallback_models", bool(self.fallback_models)),
                ("decision", self.decision is not None),
                ("http_client", self.http_client is not None),
                ("api", self.api != "auto"),
            ):
                if bad:
                    raise ValueError(
                        f"provider='claude-code' runs a local CLI and does not accept {name}"
                    )
        if self.provider == "anthropic":
            if self.fallback_models:
                raise ValueError("provider='anthropic' does not support fallback_models")
            if self.http_client is not None:
                raise ValueError(
                    "provider='anthropic' uses the Anthropic SDK; inject AnthropicLlm(client=...)"
                )
        if self.provider == "openai-compatible" and not self.base_url:
            raise ValueError("openai-compatible requires an explicit base_url")
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
        if self.provider == "openai-compatible":
            return GenericOpenAILlm(
                model=self.model,
                api_key=self.api_key,
                base_url=base_url,
                http_client=self.http_client,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
                api="chat_completions" if self.api == "auto" else self.api,
                fallback_models=self.fallback_models,
                user_agent=user_agent,
                extra_body=self.extra_body,
                extra_headers=self.extra_headers,
            )
        if self.provider == "anthropic":
            return AnthropicLlm(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url or "",
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        if self.provider == "claude-code":
            return ClaudeCodeLlm(model=self.model, timeout=self.timeout)
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

    def get_capabilities(self) -> ModelCapabilities | None:
        return self._get_backend().get_capabilities()

    async def aclose(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()
            self._backend = None


# Alias internos que conservan compatibilidad con tests y utilidades existentes.
_build_request_body = build_chat_body
_build_responses_body = build_responses_body

__all__ = ["ModelRouter"]
