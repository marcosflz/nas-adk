from __future__ import annotations

import os
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import Field, PrivateAttr, field_validator

from nas.adk.models.openai_compat import (
    DEFAULT_USER_AGENT,
    OpenAICompatibleLlm,
    env_float,
    env_int,
    parse_models,
)


class OpenCodeLlm(OpenAICompatibleLlm):
    """Proveedor de generación OpenCode Go.

    OpenCode solo genera texto: no implementa decisiones estructuradas. Para
    ``Decisor`` usa ``provider="openrouter"`` o ``provider="typesafe"``.
    """

    provider: Literal["opencode"] = "opencode"
    model: str = Field(
        default_factory=lambda: os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),
        min_length=1,
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENCODE_API_KEY", ""),
        exclude=True,
        repr=False,
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENCODE_BASE_URL", "https://opencode.ai/inference/go"
        )
    )
    api: Literal["auto", "chat_completions", "responses"] = Field(
        default_factory=lambda: _env_api()
    )
    fallback_models: tuple[str, ...] = Field(
        default_factory=lambda: parse_models(os.getenv("OPENCODE_FALLBACK_MODELS"))
    )
    session_id: str | None = Field(
        default_factory=lambda: os.getenv("OPENCODE_SESSION_ID") or None
    )
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = Field(
        default_factory=lambda: env_float("OPENCODE_TIMEOUT", 120.0), gt=0
    )
    max_retries: int = Field(
        default_factory=lambda: env_int("OPENCODE_MAX_RETRIES", 2), ge=0
    )
    backoff_base: float = Field(
        default_factory=lambda: env_float("OPENCODE_BACKOFF", 0.5), ge=0
    )

    supports_streaming: ClassVar[bool] = True
    chat_path: ClassVar[str] = "/openai/v1/chat/completions"
    responses_path: ClassVar[str] = "/openai/v1/responses"

    _session_id: str = PrivateAttr(default_factory=lambda: str(uuid4()))

    def _url(self, api: Literal["chat_completions", "responses"]) -> str:
        # Zen (modelos gratuitos, sin suscripción Go) expone /zen/v1/chat/completions,
        # mientras que Go usa /openai/v1/... Con base ".../zen" o ".../zen/v1" usamos
        # la ruta de Zen; el resto sigue igual.
        base = self.base_url.rstrip("/")
        if "/zen" in urlparse(base).path:
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            path = "/v1/chat/completions" if api == "chat_completions" else "/v1/responses"
            return f"{base}{path}"
        return super()._url(api)

    @field_validator("api", mode="before")
    @classmethod
    def _validate_api(cls, value: Any) -> Any:
        if value in (None, ""):
            return "auto"
        if value not in {"auto", "chat_completions", "responses"}:
            raise ValueError("OPENCODE_API must be auto, chat_completions or responses")
        return value

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        headers = super()._headers(session_id)
        headers["x-opencode-session"] = session_id or self.session_id or self._session_id
        return headers

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "OpenCode is a generation provider; use provider='openrouter' "
            "or provider='typesafe' for Decisor"
        )


def _env_api() -> Literal["auto", "chat_completions", "responses"]:
    value = os.getenv("OPENCODE_API", "auto")
    if value not in {"auto", "chat_completions", "responses"}:
        return "auto"
    return value  # type: ignore[return-value]


__all__ = ["OpenCodeLlm"]
