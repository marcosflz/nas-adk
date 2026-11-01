"""Personal agent: persistent SQLite memory, offline demo and capture evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from nas.adk import Agent, Runner
from nas.adk.cli import run_chat
from nas.adk.memory import MemoryExtractor, MemoryPolicy, SQLiteMemoryService, memory_tools
from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse
from nas.adk.sessions import SQLiteSessionService


class DemoLlm(BaseLlm):
    """Deterministic demo only; does not evaluate a real model's extraction quality."""

    model: str = "offline-memory-demo"

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        if request.response_format:
            source = json.loads(request.messages[-1]["content"])["source"]
            memories = []
            if "Me llamo Carlos" in source:
                memories.append({"key": "profile.name", "value": "Carlos", "category": "profile",
                                 "evidence": "Me llamo Carlos"})
            return LlmResponse(content=json.dumps({"memories": memories}))
        user = next(m["content"] for m in reversed(request.messages) if m["role"] == "user")
        if "Me llamo Carlos" in user:
            return LlmResponse(content="Encantado, Carlos.")
        for message in request.messages:
            if message.get("name") == "nas_memory":
                for memory in json.loads(message["content"]):
                    if memory["key"] == "profile.name":
                        return LlmResponse(content=f"Te llamas {memory['value']}.")
        return LlmResponse(content="No tengo tu nombre en la memoria.")


async def evaluate(model: BaseLlm) -> int:
    """Small runnable quality gate; does not save the evaluation's personal data."""
    extractor, policy = MemoryExtractor(model), MemoryPolicy()
    cases = json.loads(Path(__file__).with_name("evaluation.json").read_text(encoding="utf-8"))
    passed = false_positives = 0
    for case in cases:
        request = extractor.prepare_request(case["message"], [])
        try:
            response = await asyncio.wait_for(model.generate_content_async(request), 30)
            extraction = extractor.parse_response(response)
            keys = {m.key for m in extraction.memories if policy.accepts(m, case["message"])}
            expected = set(case["expected_keys"])
            passed += keys == expected
            false_positives += len(keys - expected)
        except Exception:
            print("Caso fallido: respuesta no válida o timeout (sin mostrar datos).")
    print(f"Captura: {passed}/{len(cases)} casos; falsos positivos: {false_positives}")
    return 0 if passed == len(cases) else 1


async def main(args: argparse.Namespace) -> int:
    model: BaseLlm
    if args.demo:
        model = DemoLlm()
    else:
        model = ModelRouter(
            provider="opencode",
            base_url=args.base_url,
            api_key=os.getenv("OPENCODE_API_KEY", ""),
            model=args.model,
            api=args.api,
        )
    try:
        if args.evaluate:
            return await evaluate(model)
        agent = Agent(
            name="personal", model=model, tools=memory_tools(),
            instruction=(
                "Eres un asistente personal. Usa los recuerdos solo cuando sean relevantes. "
                "Para peticiones explícitas de recordar u olvidar usa las herramientas y "
                "confirma únicamente si tuvieron éxito. No inventes datos ni guardes secretos."
            ),
        )
        policy = MemoryPolicy(capture="manual" if args.manual else "auto")
        extractor = None if args.manual else MemoryExtractor(model)
        with closing(SQLiteMemoryService(args.db)) as memory, closing(SQLiteSessionService(args.db)) as sessions:
            if args.demo:
                if args.manual:
                    raise ValueError("The offline demo requires automatic capture")
                runner = Runner(
                    app_name="memory_demo", agent=agent, session_service=sessions,
                    memory_service=memory, memory_policy=policy, memory_extractor=extractor,
                )
                events = [e async for e in runner.run_async(args.user_id, str(uuid4()), "Me llamo Carlos")]
                assert events[-1].metrics["memory"]["capture_status"] == "completed"
            else:
                await run_chat(
                    agent, sessions, app_name="personal", user_id=args.user_id,
                    session_id=args.session_id, memory_service=memory,
                    memory_policy=policy, memory_extractor=extractor,
                )
                return 0
        # Demonstrate a restart and a DIFFERENT conversation with the same identity.
        with closing(SQLiteMemoryService(args.db)) as memory, closing(SQLiteSessionService(args.db)) as sessions:
            runner = Runner(
                app_name="memory_demo", agent=agent, session_service=sessions,
                memory_service=memory, memory_policy=MemoryPolicy(),
            )
            events = [e async for e in runner.run_async(args.user_id, str(uuid4()), "¿Cómo me llamo?")]
            assert events[-1].status == "completed" and events[-1].content == "Te llamas Carlos."
            print(events[-1].content)
            print("OK: memoria recuperada tras reabrir SQLite en otra sesión.")
            return 0
    finally:
        await model.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="agent-memory.db")
    parser.add_argument("--user-id", default="local_user")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--model", default=os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go"),
    )
    parser.add_argument(
        "--api",
        choices=("auto", "chat_completions", "responses"),
        default=os.getenv("OPENCODE_API", "auto"),
    )
    parser.add_argument("--manual", action="store_true", help="Disable automatic extraction")
    parser.add_argument("--demo", action="store_true", help="Deterministic demo, no network calls")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate capture with the selected model")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
