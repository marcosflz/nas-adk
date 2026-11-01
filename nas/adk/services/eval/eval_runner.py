from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nas.adk import Agent, Runner
from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)


class EvalCase:
    """Caso de evaluación individual.

    Args:
        name: Nombre identificativo del caso.
        input: Entrada que se enviará al agente.
        expected: Texto esperado en la respuesta (para scoring automático).
        rubric: Rúbrica para evaluación por LLM (opcional).
    """

    def __init__(
        self,
        name: str,
        input: str,
        expected: str | None = None,
        rubric: str | None = None,
    ) -> None:
        self.name = name
        self.input = input
        self.expected = expected
        self.rubric = rubric


class EvalResult:
    def __init__(
        self,
        name: str,
        input: str,
        output: str,
        score: float,
        passed: bool,
        error: str | None = None,
        details: str | None = None,
        status: str = "completed",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.input = input
        self.output = output
        self.score = score
        self.passed = passed
        self.error = error
        self.details = details
        self.status, self.metrics = status, metrics or {}


class EvalRunner:
    """Ejecuta casos de evaluación contra un agente y genera un reporte.

    Args:
        agent: Agente a evaluar.
        evaluator_agent: Agente opcional que puntúa las respuestas (LLM-as-judge).
    """

    def __init__(
        self,
        agent: Agent,
        evaluator_agent: Agent | None = None,
    ) -> None:
        self._agent = agent
        self._evaluator = evaluator_agent

    def _score_exact(self, output: str, expected: str) -> float:
        return 1.0 if output.strip() == expected.strip() else 0.0

    def _score_substring(self, output: str, expected: str) -> float:
        return 1.0 if expected.lower() in output.lower() else 0.0

    async def _score_llm(self, case: EvalCase, output: str) -> tuple[float, str]:
        if self._evaluator is None:
            return 0.0, "no evaluator"
        rubric = case.rubric or f"The response should match or relate to: {case.expected}"
        prompt = (
            f"Rate the following response from 0.0 to 1.0 based on this rubric:\n"
            f"Rubric: {rubric}\n\n"
            f"Input: {case.input}\n"
            f"Response: {output}\n\n"
            f'Return only a JSON object: {{"score": 0.0-1.0, "reason": "..."}}'
        )
        svc = InMemorySessionService()
        runner = Runner(
            app_name="eval",
            agent=self._evaluator,
            session_service=svc,
        )
        final_text = ""
        async for event in runner.run_async(
            user_id="eval", session_id=str(uuid4()), message=prompt
        ):
            if event.final:
                if event.status != "completed":
                    return 0.0, "Evaluator did not complete"
                final_text = event.content or ""
        try:
            data = json.loads(final_text)
            score = float(data.get("score", 0))
            if not math.isfinite(score) or not 0 <= score <= 1:
                return 0.0, "Invalid evaluator score"
            return score, data.get("reason", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0.0, "parse error"

    async def run(
        self,
        cases: list[EvalCase],
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        svc = InMemorySessionService()

        for case in cases:
            started = time.monotonic()
            terminal = None
            try:
                runner = Runner(
                    app_name="eval",
                    agent=self._agent,
                    session_service=svc,
                )
                final_text = ""
                async for event in runner.run_async(
                    user_id="eval_user",
                    session_id=str(uuid4()),
                    message=case.input,
                ):
                    if event.final:
                        terminal = event
                        final_text = event.content or ""

                if terminal is None or terminal.status != "completed":
                    raise RuntimeError(final_text or "No completed terminal response")
                if case.rubric and self._evaluator is None:
                    raise ValueError("A rubric requires an evaluator_agent")

                if (case.expected is not None or case.rubric) and self._evaluator:
                    score, reason = await self._score_llm(case, final_text)
                    results.append(
                        EvalResult(
                            name=case.name,
                            input=case.input,
                            output=final_text,
                            score=score,
                            passed=score >= 0.5,
                            details=reason,
                        )
                    )
                elif case.expected is not None:
                    score = self._score_substring(final_text, case.expected)
                    results.append(
                        EvalResult(
                            name=case.name,
                            input=case.input,
                            output=final_text,
                            score=score,
                            passed=score >= 0.5,
                        )
                    )
                else:
                    results.append(
                        EvalResult(
                            name=case.name,
                            input=case.input,
                            output=final_text,
                            score=0.0,
                            passed=True,
                        )
                    )
            except Exception as e:
                results.append(
                    EvalResult(
                        name=case.name,
                        input=case.input,
                        output="",
                        score=0.0,
                        passed=False,
                        error=str(e),
                    )
                )
            results[-1].status = (
                terminal.status if terminal is not None and terminal.status else "error"
            )
            results[-1].metrics = dict(terminal.metrics) if terminal is not None else {}
            results[-1].metrics["eval_seconds"] = time.monotonic() - started

        return results

    @staticmethod
    def report(results: list[EvalResult]) -> str:
        """Genera un reporte legible con resultados de evaluación."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = sum(r.score for r in results) / total if total else 0
        lines = [
            "=== Eval Report ===",
            f"Passed: {passed}/{total}",
            f"Avg score: {avg_score:.2f}",
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name} (score={r.score:.2f})")
            if not r.passed and r.details:
                lines.append(f"  reason: {r.details}")
            if r.error:
                lines.append(f"  error: {r.error}")
        return "\n".join(lines)
