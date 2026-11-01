from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field, model_validator

from nas.adk.agents.llm_agent import LlmAgent
from nas.adk.models.base_llm import LlmResponse


class Decisor(LlmAgent):
    """Evalúa una entrada con preguntas Decision, sin generar texto ni usar tools.

    El modelo debe implementar decisiones estructuradas: ``OpenRouterLlm`` o
    ``TypesafeLlm``. ``OpenCodeLlm`` solo genera texto y será rechazado con un
    ``NotImplementedError`` explicativo.
    """

    questions: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_configuration(self) -> Decisor:
        if self.tools or self.sub_agents or self.generation_config or self.output_schema is not None:
            raise ValueError("Decisor does not support tools, sub-agents, or generation options")
        self._question_definitions()
        return self

    def _question_definitions(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, question in self.questions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Decision question names must be non-empty strings")
            if isinstance(question, BaseModel):
                definition = question.model_dump(mode="json")
            elif isinstance(question, Mapping):
                definition = dict(question)
            else:
                raise ValueError(f"Question {name!r} must be a mapping or Pydantic model")
            kind = definition.get("type")
            if not isinstance(kind, str) or kind not in {"noul", "choice", "score"}:
                raise ValueError(f"Unsupported question type for {name!r}: {kind!r}")
            criteria = definition.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, Mapping) or not criteria:
                    raise ValueError(f"Choice question {name!r} requires criteria")
                definition["criteria"] = dict(criteria)
            if kind == "score":
                if not isinstance(criteria, list):
                    raise ValueError(f"Score question {name!r} requires ordered criteria")
                if not 1 <= len(criteria) <= 10:
                    raise ValueError(f"Score question {name!r} requires 1 to 10 criteria")
            existing = definition.get("instructions")
            if self.instruction:
                if isinstance(existing, str) and existing:
                    definition["instructions"] = f"{self.instruction}\n\n{existing}"
                elif existing is None:
                    definition["instructions"] = self.instruction
                else:
                    definition["instructions"] = {
                        "context": self.instruction,
                        "question": existing,
                    }
            result[name] = definition
        return result

    def _prepare_state(self, value: Any) -> str | dict[str, Any] | list[Any]:
        if self.input_schema is not None:
            state = json.loads(self.validate_input(value))
            if isinstance(state, (str, dict, list)):
                return state
            raise ValueError("Decision schema must produce text, an object, or a list")
        if isinstance(value, (str, dict, list)):
            return value
        raise ValueError("Decision state must be text, an object, or a list")

    def _validate_result(self, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ValueError("Decision model returned no answers object")
        questions = self._question_definitions()
        answers = result["answers"]
        if answers.keys() != questions.keys():
            raise ValueError("Decision model returned missing or unexpected answers")
        for name, question in questions.items():
            answer = answers[name]
            kind = question["type"]
            if not isinstance(answer, dict) or answer.get("type") != kind:
                raise ValueError(f"Decision model returned an invalid {kind} answer for {name!r}")
            value_key = {"noul": "noul", "choice": "choice", "score": "score"}[kind]
            value = answer.get(value_key)
            if kind == "choice":
                if not isinstance(value, str) or value not in question["criteria"]:
                    raise ValueError(f"Decision model returned an unknown choice for {name!r}")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Decision model returned an invalid numeric answer for {name!r}")
            elif kind == "noul" and not 0 <= value <= 1:
                raise ValueError(f"Decision probability for {name!r} must be between 0 and 1")
            elif kind == "score" and not 0 <= value <= len(question["criteria"]) - 1:
                raise ValueError(f"Decision score for {name!r} is outside its criteria")

            confidence = answer.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError(f"Decision confidence for {name!r} must be between 0 and 1")
            probabilities = answer.get("probabilities")
            if probabilities is not None and (
                not isinstance(probabilities, dict)
                or any(
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                    for probability in probabilities.values()
                )
            ):
                raise ValueError(f"Decision probabilities for {name!r} must be between 0 and 1")
        usage = result.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("Decision model returned invalid token usage")
            for key in ("input_tokens", "output_tokens"):
                count = usage.get(key)
                if count is not None and (
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                ):
                    raise ValueError("Decision model returned invalid token usage")
        return result

    async def decide_async(self, state: Any) -> dict[str, Any]:
        """Devuelve la respuesta Decision nativa y validada del modelo."""
        result = await self.model.decide_async(self._prepare_state(state), self._question_definitions())
        return self._validate_result(result)

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        ctx: Any = None,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        user_message = next((message for message in reversed(messages) if message.get("role") == "user"), None)
        if user_message is None:
            raise ValueError("Decisor requires a user message")
        result = await self.decide_async(user_message.get("content", ""))
        usage = result.get("usage") or {}
        normalized_usage = {
            target: usage[source]
            for source, target in (
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "completion_tokens"),
            )
            if isinstance(usage.get(source), int) and not isinstance(usage[source], bool)
        }
        if "prompt_tokens" in normalized_usage and "completion_tokens" in normalized_usage:
            normalized_usage["total_tokens"] = (
                normalized_usage["prompt_tokens"] + normalized_usage["completion_tokens"]
            )
        yield LlmResponse(
            content=json.dumps(result, ensure_ascii=False),
            usage=normalized_usage,
        )
