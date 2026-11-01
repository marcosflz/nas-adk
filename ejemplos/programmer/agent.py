"""Agente programador de ejemplo. Configuración en config.py; herramientas en tools.py."""

import asyncio
import os

from nas.adk import Agent, Context
from nas.adk.cli import run_chat
from nas.adk.models import ModelRouter

if __package__:
    from . import config
    from .tools import build_tools
else:
    import config
    from tools import build_tools


def programmer(name, model, tools):
    async def spawn_programmer(task, name, ctx: Context):
        child = programmer(name, model, tools)
        return await ctx.run_agent(child, task)

    return Agent(
        name=name,
        model=model,
        instruction=config.INSTRUCTION,
        generation_config=config.GENERATION_CONFIG,
        tools=[*tools, spawn_programmer],
    )


async def main():
    workspace = config.WORKSPACE.resolve()
    os.chdir(workspace)  # run_command ejecuta desde aquí
    tools = build_tools(workspace, config.ALLOW_COMMANDS)

    if config.OFFLINE:
        print("OFFLINE")
        print(await next(tool for tool in tools if tool.name == "list_files").run_async())
        return

    if not config.OPENCODE_BASE_URL or not config.OPENCODE_API_KEY:
        raise ValueError("Set OPENCODE_BASE_URL and OPENCODE_API_KEY")
    model = ModelRouter(
        provider="opencode", base_url=config.OPENCODE_BASE_URL,
        api_key=config.OPENCODE_API_KEY, model=config.MODEL, api=config.OPENCODE_API,
        fallback_models=tuple(
            item.strip() for item in config.OPENCODE_FALLBACK_MODELS.split(",") if item.strip()
        ),
        timeout=config.REQUEST_TIMEOUT,
    )
    try:
        await run_chat(programmer("programmer", model, tools))
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
