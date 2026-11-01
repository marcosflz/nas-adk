from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import aclosing
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from nas.adk import Agent, Runner
from nas.adk.cli.loader import load_agent
from nas.adk.core.events.event import Event
from nas.adk.memory import MemoryExtractor, MemoryPolicy, MemoryService
from nas.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)


def _answer(author: str, content: str) -> Group:
    return Group(
        Text.from_ansi(author, style="bold green"),
        Markdown(Text.from_ansi(content).plain, code_theme="monokai"),
    )


def _print_tool(console: Console, event: Event, details: bool) -> None:
    content = Text.from_ansi(event.content or "").plain
    # ponytail: preview 6 líneas/800 caracteres; /tools habilita la salida completa.
    preview = content if details else "\n".join(content.splitlines()[:6])[:800]
    truncated = preview != content
    title = Text.from_ansi(f"Tool: {event.author}", style="cyan")
    seconds = event.metrics.get("tool_seconds")
    if seconds is not None:
        title.append(f" · {seconds:.2f}s", style="dim")
    console.print(Panel(Text(preview), title=title, title_align="left", border_style="dim"))
    if truncated:
        console.print(Text(
            f"Resultado abreviado · event:{event.id} · /tools muestra detalles en próximos turnos",
            style="dim",
        ))


async def _chat_turn(
    runner: Runner, session_id: str, message: str, console: Console, details: bool,
    *, user_id: str = "cli_user",
) -> None:
    chunks: list[str] = []
    author = "Asistente"
    terminal: Event | None = None
    rendered_at = 0.0
    thinking = Spinner("dots", text="Pensando…", style="cyan")
    try:
        with Live(
            thinking, console=console, transient=True, refresh_per_second=8,
            auto_refresh=console.is_terminal, redirect_stdout=False, redirect_stderr=False,
        ) as live:
            async with aclosing(runner.run_async(
                user_id=user_id, session_id=session_id, message=message, stream=True,
            )) as events:
                async for event in events:
                    if event.role == "user":
                        continue
                    if event.type == "stream":
                        chunks.append(event.content or "")
                        author = event.author or author
                        now = time.monotonic()
                        # ponytail: Markdown completo a 8 Hz; render incremental si respuestas largas pesan.
                        if console.is_terminal and now - rendered_at >= 0.125:
                            live.update(_answer(author, "".join(chunks)), refresh=True)
                            rendered_at = now
                        continue
                    live.update(Text(""), refresh=True)
                    if event.type == "status":
                        if chunks:
                            console.print(_answer(author, "".join(chunks)))
                        Console(stderr=True).print(Panel(
                            Text.from_ansi(f"[{event.status}] {event.content or ''}"),
                            title="Ejecución interrumpida", border_style="red", title_align="left",
                        ))
                        chunks.clear()
                    elif event.role == "assistant":
                        content = event.content or "".join(chunks)
                        if content:
                            console.print(_answer(event.author or author, content))
                        chunks.clear()
                        for call in event.tool_calls:
                            fn = call.get("function", {})
                            args = fn.get("arguments", "")
                            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                            if not details:
                                args = " ".join(args.splitlines())
                            console.print(Text.from_ansi(
                                f"▶ Tool: {fn.get('name', '')}({args})", style="cyan",
                                no_wrap=not details, overflow="ellipsis",
                            ))
                        if not event.final:
                            live.update(Spinner("dots", text="Ejecutando tools…", style="cyan"))
                    elif event.role == "tool":
                        _print_tool(console, event, details)
                        live.update(thinking)
                    if event.final:
                        terminal = event
    finally:
        if chunks:
            console.print(_answer(author, "".join(chunks)))
            console.print(Text("Respuesta incompleta; ejecución interrumpida.", style="yellow"))
    if terminal is not None:
        stats = [f"{terminal.metrics.get('total_seconds', 0):.2f}s"]
        calls = terminal.metrics.get("model_calls", 0)
        if calls:
            stats.append(f"{calls} llamadas")
        usage = terminal.metrics.get("total_usage", terminal.usage)
        if usage:
            incoming = usage.get("prompt_tokens", usage.get("input_tokens", "—"))
            outgoing = usage.get("completion_tokens", usage.get("output_tokens", "—"))
            stats.append(f"tokens: {incoming} entrada / {outgoing} salida")
        console.print(Text(" · ".join(stats), style="dim"))
        memory_status = terminal.metrics.get("memory", {}).get("capture_status")
        if memory_status and memory_status not in {"completed", "suppressed_by_explicit_tool"}:
            console.print(Text(
                f"Memoria no guardada automáticamente: {memory_status}", style="yellow",
            ))
    console.print()


async def run_chat(
    agent: Any, session_service: InMemorySessionService | None = None,
    *, app_name: str = "cli", user_id: str = "cli_user", session_id: str | None = None,
    memory_service: MemoryService | None = None, memory_policy: MemoryPolicy | None = None,
    memory_extractor: MemoryExtractor | None = None,
) -> None:
    """Chat Rich con Markdown, streaming, tools y cierre de proveedores propios."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # Streams capturados o ya leídos pueden impedir reconfigurar.
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
        memory_service=memory_service,
        memory_policy=memory_policy,
        memory_extractor=memory_extractor,
    )
    session_id = session_id or str(uuid4())
    console = Console()
    model = getattr(agent, "model", None)
    model_name = getattr(model, "model", "workflow offline")
    api = getattr(model, "api", None)
    api_label = f" · API: {api}" if isinstance(api, str) else ""
    console.rule(Text.from_ansi(f"NAS ADK · Chat: {agent.name}", style="bold cyan"), align="left")
    console.print(Text.from_ansi(f"Modelo: {model_name}{api_label} · sesión: {session_id[:8]}", style="dim"))
    console.print(Text("/help ayuda · /reset nueva sesión · /tools detalles · /salir salir · Ctrl+C salir", style="dim"))
    console.print()
    details = False
    try:
        while True:
            try:
                user_input = console.input(Text("Tú > ", style="bold cyan")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("exit", "quit", "/salir", "/exit"):
                break
            if user_input.lower() == "/reset":
                session_id = str(uuid4())
                console.print(Text(f"Conversación reiniciada · sesión: {session_id[:8]}", style="green"))
                continue
            if user_input.lower() == "/help":
                console.print(Panel(Text(
                    "/reset  Iniciar una conversación nueva\n"
                    "/tools  Alternar resultados compactos/completos para próximos turnos\n"
                    "/salir  Terminar (también exit, quit o Ctrl+C)\n"
                    "/help   Mostrar esta ayuda",
                ), title="Comandos", title_align="left", border_style="cyan"))
                continue
            if user_input.lower() == "/tools":
                details = not details
                console.print(Text(f"Tools: {'detalles completos' if details else 'vista compacta'}", style="cyan"))
                continue
            if not user_input:
                continue
            if user_input.startswith("/"):
                console.print(Text("Comando desconocido. Usa /help.", style="yellow"))
                continue
            console.print()
            await _chat_turn(runner, session_id, user_input, console, details, user_id=user_id)
    finally:
        await agent.aclose()


async def _run_single(
    agent: Agent,
    message: str,
    session_service: InMemorySessionService,
) -> None:
    runner = Runner(
        app_name="cli",
        agent=agent,
        session_service=session_service,
    )
    streamed = False
    async for event in runner.run_async(
        user_id="cli_user",
        session_id="cli_session",
        message=message,
        stream=True,
    ):
        if event.type == "stream":
            click.echo(event.content or "", nl=False)
            streamed = True
        elif event.final and event.status != "completed":
            raise click.ClickException(event.content or event.status or "Execution failed")
        elif event.final and not streamed:
            click.echo(event.content or "")
        elif event.final:
            click.echo()
        elif event.role == "assistant" and event.type == "message":
            if streamed:
                click.echo()
            streamed = False


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("agent_path")
@click.option("--message", "-m", default="", help="Single message (non-interactive)")
def run(agent_path: str, message: str) -> None:
    """Run an agent interactively or with a single message."""
    agent = load_agent(agent_path)
    session_service = InMemorySessionService()

    async def execute() -> None:
        try:
            if message:
                await _run_single(agent, message, session_service)
            else:
                await run_chat(agent, session_service)
        finally:
            await agent.aclose_models()
            await agent.aclose()

    asyncio.run(execute())


@main.command()
@click.argument("agent_path")
@click.option("--session-db", type=click.Path(dir_okay=False), default=None,
              help="Optional SQLite history file")
def tui(agent_path: str, session_db: str | None) -> None:
    """Open an existing agent in the optional Textual interface."""
    try:
        from nas.adk.tui import AgentApp
    except ModuleNotFoundError as error:
        if error.name != "textual":
            raise
        raise click.ClickException(str(error)) from error
    AgentApp(load_agent(agent_path), session_db=session_db).run()


@main.command()
@click.argument("name")
@click.option(
    "--provider",
    type=click.Choice(["opencode", "openrouter"]),
    default="opencode",
    show_default=True,
    help="Generation provider for the scaffold",
)
def create(name: str, provider: str) -> None:
    """Scaffold a new agent project."""
    target = Path(name)
    if target.exists():
        raise click.ClickException(f"Directory {name} already exists")
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("")
    agent_py = target / "agent.py"
    if provider == "opencode":
        model_block = (
            'model = ModelRouter(\n'
            '    provider="opencode",\n'
            '    api_key=os.getenv("OPENCODE_API_KEY", ""),\n'
            '    model=os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),\n'
            '    api=os.getenv("OPENCODE_API", "auto"),\n'
            ')\n'
        )
        env_block = (
            "# OpenCode Go\n"
            "OPENCODE_API_KEY=oc_sk_...\n"
            "OPENCODE_MODEL=deepseek-v4.1-flash\n"
            "OPENCODE_BASE_URL=https://opencode.ai/inference/go\n"
            "OPENCODE_API=auto\n"
        )
    else:
        model_block = (
            'model = ModelRouter(\n'
            '    provider="openrouter",\n'
            '    api_key=os.getenv("OPENROUTER_API_KEY", ""),\n'
            '    model=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),\n'
            ')\n'
        )
        env_block = (
            "# OpenRouter\n"
            "OPENROUTER_API_KEY=sk-or-v1-...\n"
            "OPENROUTER_MODEL=openrouter/auto\n"
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
        )
    agent_py.write_text(
        f"""import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from nas.adk import Agent
from nas.adk.cli import run_chat
from nas.adk.models import ModelRouter

load_dotenv(Path(__file__).with_name(".env"))

def get_weather(city: str) -> str:
    return f"The weather in {{city}} is sunny."

{model_block}
root_agent = Agent(
    name={name!r},
    model=model,
    instruction="You are a helpful assistant. Use tools when appropriate.",
    tools=[get_weather],
)

if __name__ == "__main__":
    async def main():
        try:
            await run_chat(root_agent)
        finally:
            await root_agent.aclose_models()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nChat finalizado.")
""",
        encoding="utf-8",
    )
    env_template = target / ".env.template"
    env_template.write_text(env_block, encoding="utf-8")
    click.echo(f"Created agent project in ./{name}/")
    click.echo(f"  {name}/__init__.py")
    click.echo(f"  {name}/agent.py")
    click.echo(f"  {name}/.env.template")
    click.echo(f"\nRun it: python {agent_py}")
    click.echo(f"Or: nas-adk run {name}")


@main.command()
@click.argument("agent_path")
@click.option("--port", "-p", default=8010, help="Port", type=int)
@click.option(
    "--host", default="127.0.0.1", help="Bind host; remote access requires NAS_ADK_API_KEY"
)
def a2a(agent_path: str, port: int, host: str) -> None:
    """Start an A2A protocol server for an agent."""
    agent = load_agent(agent_path)
    from nas.adk.protocols.a2a.server import start_a2a_server

    start_a2a_server(agent, port=port, host=host)


@main.command()
@click.argument("agent_path")
@click.option(
    "--transport",
    "-t",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="MCP transport",
)
@click.option("--port", "-p", default=8011, help="Port for SSE/HTTP", type=int)
def mcp(agent_path: str, transport: str, port: int) -> None:
    """Start an MCP server exposing the agent's tools."""
    agent = load_agent(agent_path)
    from nas.adk.protocols.mcp.server import start_mcp_server

    start_mcp_server(agent, transport=transport, port=port)


@main.command()
@click.argument("agent_path")
@click.option("--cases", "-c", default="", help="JSON file with eval cases")
def eval(agent_path: str, cases: str) -> None:
    """Evaluate an agent against test cases."""
    agent = load_agent(agent_path)
    from nas.adk.services.eval import EvalCase, EvalRunner

    if cases:
        import json

        with open(cases) as f:
            data = json.load(f)
        eval_cases = [
            EvalCase(
                name=c.get("name", f"case_{i}"),
                input=c["input"],
                expected=c.get("expected"),
                rubric=c.get("rubric"),
            )
            for i, c in enumerate(data)
        ]
    else:
        click.echo("No --cases file provided. Using 2 default cases.")
        eval_cases = [
            EvalCase(name="hello", input="Say hello", expected="hello"),
            EvalCase(
                name="weather",
                input="What's the weather?",
                expected="weather",
            ),
        ]

    runner = EvalRunner(agent)

    async def execute() -> None:
        try:
            results = await runner.run(eval_cases)
            click.echo(runner.report(results))
        finally:
            await agent.aclose_models()
            await agent.aclose()

    asyncio.run(execute())
