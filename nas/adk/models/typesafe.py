from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse
from nas.adk.models.errors import (
    LlmAuthError,
    LlmError,
    LlmResponseError,
    LlmServerError,
)
from nas.adk.models.openai_compat import (
    DEFAULT_USER_AGENT,
    env_float,
    env_int,
    error_from_status,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class TypesafeAnswer(BaseModel):
    """Respuesta normalizada de TypeSafe Jev."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    type: str
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    confidence_derived: bool = False
    legend: dict[str, str] = Field(default_factory=dict)

    def to_decision_answer(self) -> dict[str, Any]:
        """Formato exacto que valida ``Decisor``."""
        answer: dict[str, Any] = {"type": self.type}
        if self.type == "noul":
            answer["noul"] = self.noul
        elif self.type == "choice":
            answer["choice"] = self.choice
        else:
            answer["score"] = self.score
        if self.confidence is not None:
            answer["confidence"] = self.confidence
        if self.probabilities:
            answer["probabilities"] = self.probabilities
        if self.legend:
            answer["legend"] = self.legend
        return answer


def _question_type(question: Mapping[str, Any]) -> str:
    kind = question.get("type")
    if kind not in {"noul", "choice", "score"}:
        raise LlmResponseError(f"Unsupported decision question type: {kind!r}")
    return str(kind)


def _normalize_answer(
    question_id: str,
    question: Mapping[str, Any],
    payload: Any,
) -> TypesafeAnswer:
    if isinstance(payload, str):
        payload = {"choice": payload}
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        payload = {"noul": float(payload)}
    if not isinstance(payload, Mapping):
        raise LlmResponseError(f"Jev returned an invalid answer for {question_id!r}")

    expected_type = _question_type(question)
    raw_type = payload.get("type")
    if raw_type is None:
        if payload.get("noul") is not None:
            answer_type = "noul"
        elif payload.get("score") is not None:
            answer_type = "score"
        elif payload.get("choice") is not None or payload.get("probabilities"):
            answer_type = "choice"
        else:
            answer_type = expected_type
    elif isinstance(raw_type, str):
        answer_type = raw_type
    else:
        raise LlmResponseError(f"Jev returned an invalid answer type for {question_id!r}")
    if answer_type != expected_type:
        raise LlmResponseError(
            f"Jev returned type {answer_type!r} for {expected_type!r} question "
            f"{question_id!r}"
        )

    probabilities: dict[str, float] = {}
    raw_probabilities = payload.get("probabilities")
    if isinstance(raw_probabilities, Mapping):
        for option, probability in raw_probabilities.items():
            parsed = _as_float(probability)
            if parsed is not None:
                probabilities[str(option)] = _clamp01(parsed)

    choice = payload.get("choice")
    choice = str(choice) if choice is not None else None
    if choice is None and probabilities:
        choice = max(probabilities, key=lambda key: probabilities[key])

    noul = _as_float(payload.get("noul"))
    if noul is not None:
        noul = _clamp01(noul)
    score = _as_float(payload.get("score"))

    confidence = _as_float(payload.get("confidence"))
    confidence_derived = False
    if confidence is None:
        if expected_type == "noul" and noul is not None:
            confidence = abs(2.0 * noul - 1.0)
            confidence_derived = True
        elif probabilities:
            confidence = max(probabilities.values())
            confidence_derived = True
    if confidence is not None:
        confidence = _clamp01(confidence)

    legend: dict[str, str] = {}
    raw_legend = payload.get("legend")
    if isinstance(raw_legend, Mapping):
        legend = {str(key): str(value) for key, value in raw_legend.items()}

    if expected_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or not criteria:
            raise LlmResponseError(f"Choice question {question_id!r} has no criteria")
        if choice not in criteria:
            raise LlmResponseError(
                f"Jev returned an unknown choice for {question_id!r}"
            )
    elif expected_type == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise LlmResponseError(f"Score question {question_id!r} has no criteria")
        if score is None:
            raise LlmResponseError(f"Jev returned no score for {question_id!r}")
        if not 0 <= score <= len(criteria) - 1:
            raise LlmResponseError(
                f"Jev returned a score outside the criteria for {question_id!r}"
            )
    elif noul is None:
        raise LlmResponseError(f"Jev returned no noul value for {question_id!r}")

    return TypesafeAnswer(
        question_id=question_id,
        type=answer_type,
        noul=noul,
        choice=choice,
        score=score,
        probabilities=probabilities,
        confidence=confidence,
        confidence_derived=confidence_derived,
        legend=legend,
    )


def _normalize_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, Mapping):
        return {}

    def _int(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    prompt = _int("input_tokens", "prompt_tokens")
    completion = _int("output_tokens", "completion_tokens")
    total = _int("total_tokens")
    result: dict[str, int] = {}
    if prompt is not None:
        result["input_tokens"] = prompt
    if completion is not None:
        result["output_tokens"] = completion
    if total is not None:
        result["total_tokens"] = total
    elif prompt is not None and completion is not None:
        result["total_tokens"] = prompt + completion
    return result


class TypesafeLlm(BaseLlm):
    """Decisor nativo de TypeSafe Jev (``/v1/systemone``)."""

    provider: str = "typesafe"
    model: str = Field(
        default_factory=lambda: os.getenv("JEV_MODEL", "jev-latest"), min_length=1
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("TYPESAFE_API_KEY", ""),
        exclude=True,
        repr=False,
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone"
        )
    )
    timeout: float = Field(
        default_factory=lambda: env_float("JEV_TIMEOUT", 5.0), gt=0
    )
    max_retries: int = Field(
        default_factory=lambda: env_int("JEV_MAX_RETRIES", 2), ge=0
    )
    backoff_base: float = Field(
        default_factory=lambda: env_float("JEV_BACKOFF", 0.5), ge=0
    )
    user_agent: str = DEFAULT_USER_AGENT
    http_client: httpx.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )

    supports_streaming: ClassVar[bool] = False

    _client: httpx.AsyncClient | None = PrivateAttr(default=None)

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

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

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LlmAuthError("TYPESAFE_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

    async def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._get_client().post(
                    self.base_url,
                    json=_jsonable(dict(payload)),
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as error:
                failure: LlmError = LlmServerError(
                    f"Jev timed out after {self.timeout:g}s",
                    retryable=True,
                    cause=error,
                )
            except httpx.HTTPError as error:
                failure = LlmServerError(
                    f"Jev network error: {type(error).__name__}",
                    retryable=True,
                    cause=error,
                )
            else:
                if response.status_code in (401, 403):
                    failure = LlmAuthError(
                        f"Jev returned HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                elif response.status_code >= 400:
                    failure = error_from_status(response, provider="Jev")
                else:
                    try:
                        data = response.json()
                    except (ValueError, json.JSONDecodeError) as error:
                        failure = LlmResponseError(
                            "Jev returned non-JSON content",
                            status_code=response.status_code,
                            retryable=True,
                            cause=error,
                        )
                    else:
                        if not isinstance(data, Mapping):
                            failure = LlmResponseError(
                                "Jev returned non-object JSON",
                                status_code=response.status_code,
                                retryable=True,
                            )
                        else:
                            return data
            if not failure.retryable or attempt >= self.max_retries:
                raise failure
            await asyncio.sleep(self.backoff_base * (2**attempt))
        raise LlmResponseError("Jev produced no response")

    def _normalize_result(
        self,
        data: Mapping[str, Any],
        questions_payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, Mapping):
            if any(
                key in data for key in ("choice", "score", "noul", "probabilities")
            ):
                raw_answers = {next(iter(questions_payload)): data}
            else:
                raise LlmResponseError("Jev returned no answers object")

        normalized: dict[str, dict[str, Any]] = {}
        for question_id, question in questions_payload.items():
            if question_id not in raw_answers:
                raise LlmResponseError(
                    f"Jev returned no answer for question {question_id!r}"
                )
            if not isinstance(question, Mapping):
                raise LlmResponseError(f"Invalid question {question_id!r}")
            answer = _normalize_answer(
                str(question_id), question, raw_answers[question_id]
            )
            normalized[str(question_id)] = answer.to_decision_answer()

        return {
            "answers": normalized,
            "usage": _normalize_usage(data.get("usage")),
        }

    async def decide_async(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        if not questions:
            raise LlmResponseError("Jev requires at least one question")
        questions_payload = _jsonable(questions)
        if not isinstance(questions_payload, dict):
            raise LlmResponseError("Jev questions must be an object")
        payload = {
            "state": _jsonable(state),
            "model": self.model,
            "questions": questions_payload,
        }
        last_error: LlmError | None = None
        for attempt in range(self.max_retries + 1):
            data = await self._post(payload)
            try:
                return self._normalize_result(data, questions_payload)
            except LlmResponseError as error:
                last_error = error
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.backoff_base * (2**attempt))
        if last_error is not None:
            raise last_error
        raise LlmResponseError("Jev produced no decision")

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError(
            "TypeSafe Jev only supports structured decisions; use it with Decisor, "
            "not as a chat model"
        )


# Alias de conveniencia para quien prefiera el nombre del producto.
JevLlm = TypesafeLlm

__all__ = ["JevLlm", "TypesafeAnswer", "TypesafeLlm"]
