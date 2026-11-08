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

Para el proveedor `anthropic` (SDK oficial, opcional):

```bash
uv sync --extra dev --extra anthropic --inexact   # o: pip install 'nas-adk[anthropic]'
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
| `anthropic` | Sí (herramientas nativas) | Estructurada | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `claude-code` | Sí (herramientas emuladas) | Estructurada | `CLAUDE_CODE_BIN`, `CLAUDE_CODE_MODEL` (opcional) |

`ModelRouter(provider="jev", ...)` se normaliza a `"typesafe"`.
`provider="opencode-go"` se normaliza a `"opencode"`.

### Claude: `anthropic` frente a `claude-code`

Son **dos proveedores explícitos con facturación distinta**. `provider="claude"` se rechaza a
propósito para que nunca se elija API o suscripción de forma implícita, y NAS **no hace fallback
automático** de uno a otro (ni a OpenRouter/OpenCode) cuando se agota una cuota.

| | `anthropic` | `claude-code` |
|---|---|---|
| Qué es | API Messages oficial (SDK `anthropic`) | Proceso local del CLI oficial de Claude Code |
| Facturación | API de pago (`ANTHROPIC_API_KEY`) | Límites y condiciones de tu plan de suscripción |
| Tool calling | Nativo (`tool_use`/`tool_result`) | **Emulado** con salida estructurada (protocolo v1) |
| Streaming de texto | Sí | No (usa la respuesta completa) |
| Streaming de argumentos | No se expone (NAS solo recibe llamadas completas) | No |
| `usage` | Real de la API | Agregado del CLI; coste siempre `None` |
| Decisiones (`Decisor`) | Salida estructurada (sin `confidence`) | Salida estructurada (sin `confidence`) |
| Modalidades | Solo texto | Solo texto |
| Servir por A2A/MCP | Sí | Solo con `allow_personal_use_only=True` (loopback + clave) |

En ambos casos **el Runner ejecuta tus funciones Python** (permisos, `before_tool_call`,
aprobaciones, límites, sesiones, memoria); el proveedor solo devuelve llamadas normalizadas. Una
decisión de decisor (`decide_async`) es salida estructurada validada por NAS, **no** una
decisión probabilística nativa como TypeSafe Jev: no incluye `confidence` ni `probabilities`.

#### `anthropic` (API)

```python
from nas.adk import Agent, Runner
from nas.adk.models import ModelRouter

model = ModelRouter(provider="anthropic", model="<id de modelo>")  # ANTHROPIC_API_KEY
```

- El modelo es obligatorio y lo eliges tú (`ANTHROPIC_MODEL`); NAS no fija nombres de modelo.
- Los reintentos (con `Retry-After`) son solo los del SDK (`ANTHROPIC_MAX_RETRIES`); el
  proveedor no añade otro bucle y una herramienta nunca se re-ejecuta por un reintento.
- `temperature` se ignora (la API actual la rechaza) y queda en `metrics["ignored_parameters"]`.
- Los bloques nativos (`thinking` con firma, `tool_use`) se guardan en `Event.response_items` y
  se reenvían intactos al continuar, también desde SQLite. No cambies de proveedor a mitad de
  una sesión con bloques `thinking`.
- Si `_truncate_messages` del Runner elimina turnos antiguos, los modelos con *preserved
  thinking* pueden rechazar el historial editado (`LlmRequestError`): sube `max_context_tokens`.
- `max_tokens` y `refusal` no son éxito: lanzan `LlmTruncatedError` / `LlmRefusalError`.

#### `claude-code` (suscripción local)

Requisitos: Claude Code instalado y autenticado **por ti** (`claude auth login`) con tu
suscripción. NAS nunca inicia login, no lee ni copia tokens y no usa endpoints privados.

```python
model = ModelRouter(provider="claude-code")  # CLAUDE_CODE_MODEL es opcional
```

- **Uso personal y local.** Las condiciones de Anthropic piden API key para productos o
  servicios y no permiten enrutar peticiones de terceros por credenciales de suscripción. Este
  proveedor es para el titular de la suscripción, en su propia máquina y en proyectos propios;
  la responsabilidad de cumplir las condiciones vigentes es tuya. Por eso `create_a2a_app` y
  `create_mcp_server` rechazan por defecto agentes que lo usen (`personal_use_only`). Para
  automatizaciones internas propias pasa `allow_personal_use_only=True`: se exige host en
  loopback y, en A2A, `api_key` (o `NAS_ADK_API_KEY`). Eso limita la superficie, pero no puede
  comprobar el origen de los prompts: no reenvíes peticiones de otras personas a estos agentes.
  Ejemplo: `ejemplos/claude-gastos/a2a_interno.py` (`--offline` no gasta cuota).
- **Modo estricto de suscripción.** Antes de invocar el CLI se ejecuta `claude auth status
  --json` y solo se acepta `apiProvider=firstParty` con `authMethod` `claude.ai`/`oauth_token`.
  Si hay variables que puedan facturar por API (`ANTHROPIC_*`, `CLAUDE_CODE_USE_*`,
  `AWS_BEARER_TOKEN_BEDROCK`) se lanza `LlmBillingModeError` (política `reject`, por defecto);
  con `CLAUDE_CODE_ENV_POLICY=scrub` se eliminan **solo del entorno del proceso hijo**
  (`os.environ` no se toca). `--bare` no se usa: nunca lee OAuth.
- **Aislamiento.** El CLI solo *decide*: `--tools ""`, `--disallowedTools "mcp__*"`,
  `--strict-mcp-config`, `--restricted`, `--safe-mode`, `--disable-slash-commands`,
  `--permission-mode dontAsk`, `--permission-prompts none`, `--no-session-persistence`, `cwd`
  temporal vacío y variables `CLAUDE_CODE_DISABLE_*`. No se usa `--dangerously-skip-permissions`,
  `--resume` ni `--continue`: NAS es la única fuente del historial y lo reenvía completo por
  stdin en cada llamada. Si el contenido supera `CLAUDE_CODE_MAX_INPUT_BYTES` se lanza un error
  (no se trunca en silencio). El evento `init` se comprueba en cada llamada y, si aparece una
  herramienta, MCP o plugin externo, se aborta con `LlmIsolationError`.
- **Qué se envía y qué se ejecuta.** A Anthropic (vía tu CLI) viaja el historial, las
  instrucciones y los esquemas de tus herramientas. Las funciones se ejecutan localmente en el
  Runner; los resultados de herramienta se marcan como datos no confiables.
- **Protocolo v1.** El CLI responde `{"protocol_version":1,"kind":"final","text":...}` o
  `{"kind":"tool_calls","calls":[{"name":...,"arguments":{...}}]}` mediante `--json-schema`. La
  raíz del esquema es un objeto plano (la API no admite `anyOf` en la raíz); cada llamada es una
  unión discriminada por herramienta o, si el esquema es demasiado grande, un esquema de
  transporte simple. **En ambos casos la validación estricta (nombre permitido, argumentos
  contra su JSON Schema, número de llamadas) la hace NAS en local.** Si la salida es inválida
  se reintenta la *inferencia* (`CLAUDE_CODE_PROTOCOL_REPAIRS`), nunca una herramienta.
- **Cuota agotada.** Un `rate_limit_event` rechazado produce `LlmQuotaError` (sin fallback). NAS
  no calcula saldo restante ni coste.
- **Streaming no soportado:** `supports_streaming=False`; `LlmAgent` usa la respuesta completa.
  El JSON del protocolo no se publica como texto del asistente.
- **Concurrencia/timeout:** `CLAUDE_CODE_MAX_CONCURRENCY` (1 por defecto) y
  `CLAUDE_CODE_TIMEOUT`. La cancelación y el timeout terminan el grupo de procesos completo.

Versión probada del CLI: **2.1.319** (mínimo exigido: 2.1.259; se comprueban las flags en
`claude --help`). Verificado contra el CLI real (2 invocaciones): el `init` reporta
`apiKeySource: none`, como única herramienta `StructuredOutput`, sin MCP y en modo `dontAsk`
(más 3 plugins *builtin* del propio CLI, aceptados); respuesta final; y petición de herramienta
con el esquema discriminado. **No** verificado en vivo: bucle completo con el resultado de la
herramienta, cuota agotada, y `--max-turns`/`--effort` (se usan solo si aparecen en
`claude --help`; `--max-turns` no aparece en 2.1.319).

Ejemplos equivalentes con la misma función Python: `ejemplos/claude-gastos/api.py` (API) y
`ejemplos/claude-gastos/suscripcion.py` (Claude Code).

Solución de problemas: `LlmProcessError: not found` → instala Claude Code o fija
`CLAUDE_CODE_BIN`; `LlmAuthError` → ejecuta tú `claude auth login`; `LlmBillingModeError` →
quita las variables citadas o usa `CLAUDE_CODE_ENV_POLICY=scrub`; `LlmIsolationError` → versión
del CLI con otro comportamiento (abre un issue con la versión); `LlmQuotaError` → espera al
reinicio de tu cuota.

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
- `ejemplos/claude-gastos`: la misma función Python con `anthropic` (API) y `claude-code` (suscripción).

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

ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=
ANTHROPIC_BASE_URL=
ANTHROPIC_TIMEOUT=120
ANTHROPIC_MAX_RETRIES=2

CLAUDE_CODE_BIN=claude
CLAUDE_CODE_MODEL=
CLAUDE_CODE_TIMEOUT=300
CLAUDE_CODE_MAX_CONCURRENCY=1
CLAUDE_CODE_ENV_POLICY=reject
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
