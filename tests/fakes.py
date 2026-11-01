from __future__ import annotations

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse


class FakeLlm(BaseLlm):
    model: str = "fake"
    response: LlmResponse = LlmResponse(content="ok")

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        return self.response
