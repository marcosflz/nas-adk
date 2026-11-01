# Programador de terminal con subagentes

Un mini agente programador, no un clon completo de OpenCode. Reutiliza `Agent`,
`Context.run_agent`, `run_chat` y `CodeExecutor`; no modifica el núcleo del ADK.

## Ejecutar

Configura el modelo en `config.py` y las credenciales OpenCode en el entorno; no hay
argumentos CLI ni es necesario guardar secretos en un `.env`:

```python
import os
from pathlib import Path

MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")
OPENCODE_API = os.getenv("OPENCODE_API", "auto")
GENERATION_CONFIG = {}   # Ej.: {"reasoning_effort": "low"} si el modelo lo admite.
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go")
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_FALLBACK_MODELS = os.getenv("OPENCODE_FALLBACK_MODELS", "")
REQUEST_TIMEOUT = 120.0  # Timeout del proveedor; no cambia los límites del Runner.
OFFLINE = False
WORKSPACE = Path("C:/proyectos/demo")  # Un proyecto existente.
ALLOW_COMMANDS = False
```

Selecciona un proyecto separado del fichero con tus claves para no enviarlas al
modelo al leer código. OpenCode requiere `OPENCODE_API_KEY` y `OPENCODE_BASE_URL`.
Sus llamadas pueden consumir tokens; TLS sigue validado.

Desde la raíz de este repositorio, con el entorno de desarrollo instalado:

```powershell
. .\.venv\Scripts\Activate.ps1
# Directamente, como en el IDE:
python ejemplos/programmer/agent.py
# También se puede ejecutar como módulo:
python -m ejemplos.programmer.agent
```

`OFFLINE=True` **solo lista archivos del workspace** y termina, sin modelo,
conexiones, edición ni delegación. El modo online usa el modelo OpenCode configurado en `MODEL`.

El chat existente ofrece `/help`, `/tools`, `/reset` y `/salir`. No se inicia al
importar el módulo. Se puede ejecutar directamente como `agent.py` o con `-m`.

## Cómo leer el código

```text
config.py
  modelo, credenciales, workspace, límites visuales y prompt del agente

agent.py
  create_programmer()   → construye un Agent con las herramientas
    spawn_programmer()  → construye un hijo y espera ctx.run_agent(hijo, tarea)
  main()                → abre/cierra el modelo y arranca el chat

tools.py
  resolve_path()        → control compartido de rutas
  build_tools()         → funciones de archivos y comandos para un workspace
```

Cada hijo recibe su tarea e instrucciones, **no el historial del padre**. Usa
la misma instancia de modelo y las mismas herramientas; puede crear otros hijos.
No es un proceso independiente ni un worker en segundo plano. Las delegaciones
y herramientas son secuenciales y los archivos del workspace se comparten.
El padre recibe el resultado final del hijo. El chat muestra la llamada de
delegación y su resultado, no un streaming de las operaciones internas del hijo.

## Herramientas

| Herramienta | Uso |
|---|---|
| `list_files(path=".", offset=0, limit=100)` | Listado recursivo paginado. |
| `read_file(path, start_line=1, limit=100)` | Fragmento con números de línea. |
| `search_text(text, path=".", limit=50)` | Búsqueda literal, sensible a mayúsculas, por línea. |
| `create_file(path, content)` | Crear, nunca sobrescribir. |
| `edit_file(path, old_text, new_text)` | Reemplazo exacto y único, sin números de línea. |
| `run_command(command, timeout=30)` | Shell opt-in, resultado con stdout, stderr y exit_code. |
| `spawn_programmer(task, name)` | Delegar a un programador creado bajo demanda. |

La edición conserva los bytes de texto no modificados, incluidos CRLF/LF. Para
reemplazos multilínea hay que respetar esos saltos. Se escribe en un temporal
del mismo directorio y se reemplaza el archivo; si cambió antes de la comprobación,
la edición se rechaza. No hay bloqueo contra procesos externos que editen a la vez.

## Seguridad y límites

- El workspace debe existir; sus archivos pueden modificarse incluso sin comandos.
- Las herramientas de archivos rechazan rutas absolutas, escapes y enlaces que
  resuelvan fuera de la raíz. Los recorridos no siguen symlinks.
- Se omiten `.git`, entornos, dependencias, cachés y nombres habituales de
  credenciales, como `.env*`, claves privadas y `.pem`. **No es un detector de
  secretos**: una clave incrustada en código puede enviarse al modelo.
- Solo texto UTF-8 de hasta 256 KiB. La búsqueda omite archivos ilegibles,
  binarios, no UTF-8 y más grandes: «sin coincidencias» no prueba su ausencia allí.
- Máximo 200 resultados/líneas por petición, salida de 12000 caracteres y
  recorridos de hasta 10000 entradas. Acota la carpeta o el rango si alcanzas límites.
- Presupuesto existente del Runner, compartido por padre e hijos: 20 llamadas
  al modelo, 10 turnos por agente, profundidad 4 y 120 segundos por ejecución.
- `/reset` reinicia la conversación, **no revierte archivos**. No hay undo automático.

Los comandos están deshabilitados por defecto. `ALLOW_COMMANDS = True` en
`agent.py` los habilita para código que confíes en ejecutar, preferiblemente
en un entorno desechable.

Esta variable autoriza comandos para toda la conversación, **sin aprobación por
comando**. Se inician en el workspace usando `cmd.exe` en Windows o `/bin/sh` en
POSIX, con timeout máximo de 30s y 4096 bytes por stream. No son un sandbox:
pueden salir de la raíz, acceder a otros archivos, red y credenciales del entorno.
La protección de rutas solo aplica a las herramientas de archivos; tampoco es
aislamiento frente a cambios maliciosos simultáneos del filesystem.

## Petición de ejemplo

> Lee AGENTS.md y revisa la función de validación. Delega a un hijo llamado
> validaciones que implemente el cambio en src/validation.py. Revisa su resultado
> y añade una comprobación pequeña. No instales dependencias ni cambies Git.
> Resume los archivos modificados y las comprobaciones realmente ejecutadas.

El modelo decide cuándo usar las herramientas. Para probar la delegación,
pídela expresamente con una tarea y rutas concretas.

## Comprobación local, sin modelo ni red

```powershell
python -m pytest tests/test_programmer.py -v
ruff check ejemplos/programmer/ tests/test_programmer.py
```

No añade TUI, persistencia, paralelismo, worktrees ni workers en segundo plano.
