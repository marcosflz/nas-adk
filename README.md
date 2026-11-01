# NAS ADK

Agent Development Kit async para construir agentes con proveedores LLM
intercambiables. Incluye:

- **OpenCode Go** para generación de texto y tool calling.
- **OpenRouter** para generación y decisiones estructuradas.
- **TypeSafe Jev** para decisiones estructuradas nativas.
- Sesiones en memoria o SQLite, memoria a largo plazo, workflows, subagentes,
  protocolos A2A/MCP, evaluación y un chat Rich.

Foundry y `edhm-sdk` fueron eliminados. `provider="foundry"` produce un error
explícito; usa `opencode`, `openrouter` o `typesafe`.

## Instalación

```bash
uv sync --extra dev --inexact
# o
pip install nas-adk
```

Para la TUI opcional:

```bash
uv sync --extra dev --extra tui --inexact
```

## Proveedores

| Proveedor | Generación | Decisión | Variables |
|---|---:|---:|---|
| `opencode` | Sí | No | `OPENCODE_API_KEY`, `OPENCODE_MODEL` |
| `openrouter` | Sí | Sí | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| `typesafe` / `jev` | No | Sí | `TYPESAFE_API_KEY`, `JEV_MODEL` |

`ModelRouter(provider="jev", ...)` se normaliza a `"typesafe"`.
`provider="opencode-go"` se normaliza a `"opencode"`.

### OpenCode Go

```python
import os
from nas.adk.models import ModelRouter

model = ModelRouter(
    provider="opencode",
    model=os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),
    api_key=os.environ["OPENCODE_API_KEY"],
    api=os.getenv("OPENCODE_API", "auto"),  # auto, chat_completions o responses
    fallback_models=("glm-5.3-flash", "qwen3.8-flash"),
)
```

- Base URL: `https://opencode.ai/inference/go`.
- Endpoints: `/openai/v1/chat/completions` y `/openai/v1/responses`.
- `api="auto"` envía modelos `gpt-*`, `grok-*` y `muse-*` a Responses; el resto a
  Chat Completions.
- `x-opencode-session` se toma de `LlmRequest.session_id`, `ModelRouter.session_id`
  o de un UUID estable por instancia.
- `usage.opencode.allowance` es el importe **consumido** por la petición, no el
  saldo restante. El fallback se activa con `402`, `403`, `404`, `429`, `5xx`,
  timeout o respuesta inválida.

OpenCode no implementa `decide_async()`. Si se usa con `Decisor`:

```text
NotImplementedError: OpenCode is a generation provider; use provider='openrouter'
or provider='typesafe' for Decisor
```

### OpenRouter

```python
import os
from nas.adk.models import DecisionOptions, ModelRouter

model = ModelRouter(
    provider="openrouter",
    model=os.environ["OPENROUTER_MODEL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
    app_title=os.getenv("OPENROUTER_APP_TITLE", "NAS ADK"),
    decision=DecisionOptions(strict=True, require_parameters=True),
)
```

- Base URL: `https://openrouter.ai/api/v1`.
- Chat Completions por defecto; `api="responses"` es opcional.
- `extra_body` permite routing de proveedores; `DecisionOptions.require_parameters`
  añade `{"provider": {"require_parameters": true}}`.

## Generación y streaming

```python
import asyncio
from nas.adk import Agent, Runner
from nas.adk.models import ModelRouter

async def main() -> None:
    model = ModelRouter(
        provider="opencode",
        model="deepseek-v4.1-flash",
        api_key="oc_sk_...",
    )
    agent = Agent(name="assistant", model=model, instruction="Responde en español.")
    try:
        async for event in Runner(agent=agent).run_async("user", "session", "Hola", stream=True):
            if event.type == "stream":
                print(event.content or "", end="", flush=True)
            elif event.final and event.status != "completed":
                raise RuntimeError(event.content)
    finally:
        await model.aclose()

asyncio.run(main())
```

`Agent.aclose()` cierra subagentes, no modelos compartidos. Usa
`Agent.aclose_models()` solo si el agente es dueño del modelo. Un `http_client`
inyectado no se cierra.

## Decisiones estructuradas

`Decisor` conserva su API y valida las respuestas. Solo acepta modelos con
decisión nativa o estructurada.

### TypeSafe Jev

```python
import os
from nas.adk import Decisor, Runner
from nas.adk.models import ModelRouter

model = ModelRouter(
    provider="typesafe",
    model=os.getenv("JEV_MODEL", "jev-latest"),
    api_key=os.environ["TYPESAFE_API_KEY"],
    base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
)

decisor = Decisor(
    name="triage",
    model=model,
    instruction="Clasifica con criterios conservadores.",
    questions={
        "team": {
            "type": "choice",
            "instructions": "¿Qué equipo debe atender la solicitud?",
            "criteria": {
                "technical": "Errores y fallos",
                "billing": "Cobros y facturas",
            },
        },
        "urgent": {"type": "noul", "instructions": "¿Requiere atención urgente?"},
    },
)
```

Jev también puede consumirse a través de OpenCode usando
`base_url="https://opencode.ai/inference/systemone/v1/systemone"` y una clave de
OpenCode; el proveedor sigue siendo `typesafe` y no envía `x-opencode-session`.

TypeSafe devuelve `{model, answers, usage}`. El adaptador:

- normaliza `choice`, `score` y `noul`;
- deriva `confidence = abs(2 * noul - 1)` cuando Jev no la envía;
- elige `choice` con la mayor probabilidad si falta el valor explícito;
- valida `score` contra la longitud de `criteria`;
- ignora respuestas extra y falla si falta una pregunta solicitada;
- no envía `x-opencode-session`.

### OpenRouter

```python
from nas.adk.models import DecisionOptions, ModelRouter

model = ModelRouter(
    provider="openrouter",
    model="<model-slug>",
    api_key="sk-or-v1-...",
    decision=DecisionOptions(
        strict=True,
        require_parameters=True,
        max_retries=2,
    ),
)
```

OpenRouter recibe un JSON Schema con exactamente las preguntas solicitadas. Si el
modelo no acepta `json_schema`, se reintenta con `json_object`. La respuesta se
parsea tolerando fences Markdown y se valida contra las preguntas.

## Sesiones

`LlmRequest.session_id` se propaga desde `Context.session.id`, lo que permite que
OpenCode reciba `x-opencode-session` sin configuración adicional.

```python
from nas.adk.sessions import SQLiteSessionService

sessions = SQLiteSessionService("sessions.db")
try:
    runner = Runner(agent=agent, session_service=sessions)
    # ...
finally:
    sessions.close()
```

## Workflows

```python
from nas.adk import Workflow, node

@node
async def fetch(ctx, topic: str) -> str:
    return f"Data about {topic}"

@node
async def analyze(ctx, data: str) -> str:
    return f"Analysis: {data}"

workflow = Workflow(name="pipeline", edges=[
    ("START", fetch),
    (fetch, analyze),
])
```

Los workflows son DAGs: se rechazan ciclos, nombres desconocidos y nodos no
alcanzables desde `START`. Las ramas concurrentes requieren `parallel_safe=True`.

## Subagentes

```python
from pydantic import BaseModel
from nas.adk import Agent

class ResearchInput(BaseModel):
    topic: str

class ResearchOutput(BaseModel):
    summary: str

researcher = Agent(
    name="researcher",
    model=model,
    instruction="Investiga el tema.",
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
)

coordinator = Agent(
    name="coordinator",
    model=model,
    instruction="Delega la investigación en researcher.",
    sub_agents=[researcher],
)
```

Cada subagente recibe su tarea, no el historial del padre. El resultado final
vuelve al padre mediante la Task API.

## A2A y MCP

```bash
nas-adk a2a my_agent --port 8010
nas-adk mcp my_agent --transport stdio
```

A2A implementa un subconjunto legacy (`tasks/send`, `tasks/get`), con tareas
locales y efímeras. MCP expone las tools del agente y permite usar servidores MCP
externos.

## Memoria SQLite

```python
from nas.adk import Agent, Runner
from nas.adk.memory import (
    MemoryExtractor, MemoryPolicy, SQLiteMemoryService, memory_tools,
)
from nas.adk.sessions import SQLiteSessionService

memory = SQLiteMemoryService("agent-memory.db")
sessions = SQLiteSessionService("agent-memory.db")
agent = Agent(
    name="personal", model=model, tools=memory_tools(),
    instruction="No inventes recuerdos ni guardes secretos.",
)
runner = Runner(
    app_name="personal", agent=agent, session_service=sessions,
    memory_service=memory, memory_policy=MemoryPolicy(capture="auto"),
    memory_extractor=MemoryExtractor(model),
)
```

El historial y los recuerdos son independientes aunque compartan archivo. La
captura automática añade como máximo una llamada estructurada por turno raíz y
exige evidencia literal del mensaje actual. El allowlist no es un detector de
secretos; evalúa el modelo antes de habilitarlo con usuarios reales.

## CLI

```bash
nas-adk run my_agent/agent.py
nas-adk run my_agent/agent.py --message "Hola"
nas-adk create my_agent --provider opencode
nas-adk create my_agent --provider openrouter
nas-adk a2a my_agent/agent.py --port 8010
nas-adk mcp my_agent/agent.py
nas-adk eval my_agent/agent.py --cases casos.json
```

`create` genera `agent.py`, `__init__.py` y `.env.template` sin secretos reales.

## TUI

```bash
uv sync --extra dev --extra tui --inexact
python -m ejemplos.tui
```

El chat Rich y la TUI soportan streaming, tools, subagentes, `/reset`, `/tools`
y sesiones SQLite. Para construir un EXE en Windows:

```bash
pip install "nas-adk[tui,build]"
python -m nas.adk.tui.build app.py --name MiAsistente
```

Un ejecutable no oculta claves: no incluyas `.env`, bases de datos ni secretos en
`assets/`.

## Ejemplos

- `ejemplos/investigador-terminal`: chat OpenCode + workflow offline de datos ficticios.
- `ejemplos/programmer`: agente programador con subagentes y workspace.
- `ejemplos/decisor-individual`: `Decisor` con TypeSafe Jev.
- `ejemplos/decisor-subagente`: chat OpenCode + decisor TypeSafe.
- `ejemplos/decisor-openrouter`: decisor OpenRouter con JSON Schema.
- `ejemplos/memoria-sqlite`: memoria persistente con OpenCode o demo offline.
- `ejemplos/tui`: aplicación Textual con OpenCode.

## Variables de entorno

```dotenv
NAS_MODEL_PROVIDER=opencode

OPENCODE_API_KEY=
OPENCODE_MODEL=deepseek-v4.1-flash
OPENCODE_BASE_URL=https://opencode.ai/inference/go
OPENCODE_API=auto
OPENCODE_FALLBACK_MODELS=
OPENCODE_TIMEOUT=120
OPENCODE_MAX_RETRIES=2

OPENROUTER_API_KEY=
OPENROUTER_MODEL=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=
OPENROUTER_APP_TITLE=NAS ADK
OPENROUTER_FALLBACK_MODELS=

TYPESAFE_API_KEY=
JEV_MODEL=jev-latest
JEV_BASE_URL=https://api.typesafe.ai/v1/systemone
JEV_TIMEOUT=5
JEV_MAX_RETRIES=2
```

## Tests y release

```bash
uv run ruff check nas/adk/ nas/adk/tui/ tests/
uv run mypy nas/adk/
uv run pytest tests/ -q
uv run python -m build
uv run python -m twine check dist/*
```

Todos los tests de proveedores usan `httpx.MockTransport`; no hacen llamadas de red.

GitHub Actions:

- `.github/workflows/ci.yml`: ruff, mypy, pytest en Python 3.10, 3.12 y 3.13,
  build, `twine check` y artifact.
- `.github/workflows/release.yml`: detecta el bump con Commitizen, publica un
  GitHub Release con wheel y sdist, y sube el artifact.
- `scripts/ci/bump_version.sh`: lógica reutilizable del bump.
- No se usa GitHub Packages: el wheel y el sdist se adjuntan al Release.

Si `main` tiene branch protection, configura un PAT `RELEASE_TOKEN` o publica
desde tags.

## Migración desde Foundry

- `provider="foundry"`, `endpoint`, `credential`, `manager`, `api_key_header` y
  `ca_bundle` ya no existen.
- Sustituye el modelo por `provider="opencode"` (generación) o
  `provider="typesafe"`/`provider="openrouter"` (decisión).
- `edhm-sdk` ya no es dependencia.
- `azure-pipelines.yml` fue reemplazado por GitHub Actions.
