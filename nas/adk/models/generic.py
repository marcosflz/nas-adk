from __future__ import annotations

from typing import Any, ClassVar, Literal

from nas.adk.models.openai_compat import OpenAICompatibleLlm


class GenericOpenAILlm(OpenAICompatibleLlm):
    """Cualquier endpoint compatible con OpenAI Chat Completions.

    Cubre OpenAI, Gemini (endpoint OpenAI-compatible), DeepSeek, LM Studio,
    Ollama o un gateway propio: basta con ``base_url`` (y ``api_key`` si el
    servidor la exige). La API por defecto es Chat Completions porque es la que
    ofrecen casi todos; ``api="responses"`` es opcional.
    """

    provider: Literal["openai-compatible"] = "openai-compatible"
    api: Literal["auto", "chat_completions", "responses"] = "chat_completions"

    supports_streaming: ClassVar[bool] = True

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        # Los servidores locales no piden clave: solo se envía si existe.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "openai-compatible is a generation provider; use provider='openrouter' "
            "or provider='typesafe' for Decisor"
        )



__all__ = ["GenericOpenAILlm"]
