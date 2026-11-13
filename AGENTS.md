## Project: nas-adk

ADK Python async con proveedores intercambiables: OpenCode Go y OpenRouter para
generación, y OpenRouter o TypeSafe Jev para decisiones estructuradas.

### Commands
- `source .venv/bin/activate` antes de cualquier operación
- Instalar en dev: `uv sync --extra dev --extra tui --extra a2a --extra anthropic --inexact`
- Tests: `python -m pytest tests/ -q`
- Lint: `ruff check nas/adk/ nas/adk/tui/ tests/`
- Type check: `mypy nas/adk/`
- Paquete: `python -m build && python -m twine check dist/*`

### Architecture
```
nas/adk/
├── core/            → Event, BaseCallback, CallbackManager
├── agents/          → LlmAgent, BaseAgent, Context, Decisor
├── models/          → ModelRouter, OpenCodeLlm, OpenRouterLlm, TypesafeLlm, AnthropicLlm, ClaudeCodeLlm
├── tools/           → BaseTool, FunctionTool, SubAgentTool
├── runners/         → Runner (tool-calling loop, streaming)
├── sessions/        → Session, InMemorySessionService, SQLite
├── memory/          → InMemoryMemoryService, SQLiteMemoryService
├── orchestration/   → Workflow (grafos), Planner (ReAct)
├── protocols/       → A2A, MCP (Model Context Protocol)
├── services/        → Artifacts, CodeExecutor, EvalRunner
├── ui/              → Intelligent UI: present_openui/prompt_openui/suggest_followups, TextCallRecovery, reglas de prompt
└── cli/             → Chat Rich; comandos click (run, create, a2a, mcp, eval)
```

`nas/adk/tui/` es el módulo hermano opcional para aplicaciones Textual, chat y actividad
de agentes. Instalar con `uv sync --extra dev --extra tui --inexact`; demo offline:
`python -m ejemplos.tui`.

### Providers
- `provider="opencode"`: generación con OpenCode Go. No implementa `decide_async()`.
- `provider="openrouter"`: generación y decisiones con JSON Schema.
- `provider="typesafe"` (alias `"jev"`): decisiones nativas TypeSafe Jev.
- `provider="anthropic"`: API oficial (SDK opcional `nas-adk[anthropic]`), herramientas nativas.
- `provider="claude-code"`: CLI local de Claude Code (suscripción, solo uso personal), herramientas
  emuladas; no se sirve por A2A/MCP. Sin fallback automático entre ambos ni a otros proveedores.
- `provider="foundry"` ya no existe.

### Conventions
- Paquete regular `nas.adk`: `from nas.adk import ...`
- Distribución `nas-adk` con Hatchling; empaquetar `nas/adk`
- API pública por área: `from nas.adk.agents import Agent`, `from nas.adk.runners import Runner`
- Async-first: todos los métodos de ejecución son async
- Pydantic v2 para modelos de datos
- httpx para llamadas HTTP a proveedores
- Ruff + mypy strict para linting/type checking
- Tests sin red: usar `httpx.MockTransport`
- Release en GitHub: `.github/workflows/release.yml` y `scripts/ci/bump_version.sh`
