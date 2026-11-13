"""Reglas de prompt de la UI generativa y de las preguntas de seguimiento.

Valores por defecto: la aplicación puede sustituir `ui_rules` y `followup_rules`.
"""

from __future__ import annotations

from collections.abc import Collection

from .openui import (
    OPENUI_PRESENT,
    OPENUI_PROMPT,
    OPENUI_SPEC,
    SUGGEST_FOLLOWUPS,
)
from .textcalls import RECOVERABLE_TOOLS

UI_RULES = """\
Interfaz generativa (present_openui): úsala solo cuando aporte algo que el texto no puede dar.
- Úsala para: comparar varias opciones con varios atributos (tabla o tarjetas), series numéricas
  o evolución en el tiempo (gráfico), proporciones de un total, itinerarios o pasos, y paneles
  con varias cifras clave.
- No la uses para: saludos o charla, respuestas cortas o de una sola cifra, explicaciones
  conceptuales, opiniones, código, ni cuando no tengas datos reales que mostrar. Entonces responde
  solo con texto.
- Como mucho una interfaz por respuesta. La prosa y las citas van fuera de la interfaz; la
  interfaz no repite el texto, lo complementa.
- Gráficos: solo con datos reales (de las herramientas o del usuario) y al menos 3 puntos.
  Líneas o áreas para la evolución en el tiempo, barras para comparar categorías, circular solo
  para partes de un total con 6 partes o menos. Etiquetas cortas, títulos descriptivos y las
  unidades en los ejes o en el título (€, %, millones…). Ordena las categorías de forma útil
  (cronológica o de mayor a menor). Si importan los valores exactos, añade también una tabla.
- Agrupa: una Card con un título claro, el gráfico o la tabla y, si ayuda, 2 a 4 métricas
  destacadas. Nada de tarjetas vacías ni de texto de relleno.
Si necesitas datos del usuario antes de continuar, usa prompt_openui y espera su respuesta. Si
dudas de la sintaxis, mejor una respuesta de texto correcta.
"""

UI_SPEC_HINT = "Si no conoces la sintaxis, llama antes a openui_spec (perfil minimal, data, forms, lists o full).\n"

UI_REQUESTED_RULES = """\
El usuario ha pedido expresamente una respuesta con interfaz. Antes de terminar DEBES llamar a
present_openui exactamente una vez con un programa OpenUI válido que presente lo esencial de la
respuesta (tarjeta, tabla, lista, pasos o gráfico, según encaje). El argumento `ui` es solo el
programa (empieza por `root = `), sin bloque de código ni prosa. Si devuelve un error, corrígelo y
vuelve a llamar. Escribe además una respuesta breve en texto con sus citas.
"""

TOOL_CALL_RULE = """\
{names}: son herramientas y se invocan siempre con una llamada de herramienta real. Nunca escribas
su nombre ni sus argumentos en el texto de la respuesta; escribirlos no muestra nada al usuario.
"""

UI_SPEC_HEADER = """\
Especificación de OpenUI Lang (perfil {profile}). Describe solo el lenguaje del argumento `ui` de
las herramientas: donde diga que toda tu respuesta debe ser OpenUI, se refiere a ese argumento, y tu
respuesta de texto sigue siendo prosa. Los argumentos de los componentes son POSICIONALES: escribe
`Card([a, b])` y `LineChart(labels, series, "linear", "Mes", "Precio")`, nunca
`Card(children: [a, b])` ni `variant: "linear"`.

{spec}
"""

FOLLOWUP_RULES = """\
Al terminar la respuesta, llama a la herramienta suggest_followups con 3 a 5 preguntas cortas.
Nunca escribas esas preguntas en el texto. Tras un saludo o una charla breve no hace falta.
"""


def _ending(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def ui_instructions(
    tool_names: Collection[str],
    *,
    spec: str | None = None,
    profile: str = "minimal",
    requested: bool = False,
    ui_rules: str | None = None,
    followup_rules: str | None = None,
) -> str:
    """Fragmento del prompt del sistema para las herramientas de UI que lleva el agente.

    `spec` es la especificación del perfil ya resuelta; con ella en el prompt no hace falta
    `openui_spec`. `requested` fuerza a presentar una interfaz antes de terminar.
    Devuelve cadena vacía si el agente no lleva ninguna herramienta de UI.
    """
    names = set(tool_names)
    parts: list[str] = []
    if {OPENUI_PRESENT, OPENUI_PROMPT} & names:
        inline = bool(spec and spec.strip())
        parts.append(_ending(ui_rules if ui_rules and ui_rules.strip() else UI_RULES))
        if not inline and OPENUI_SPEC in names:
            parts.append(UI_SPEC_HINT)
        if OPENUI_PRESENT in names:
            if requested:
                parts.append(UI_REQUESTED_RULES)
            if inline and spec is not None:
                parts.append(UI_SPEC_HEADER.format(profile=profile, spec=spec.strip()))
    if SUGGEST_FOLLOWUPS in names:
        rules = followup_rules if followup_rules and followup_rules.strip() else FOLLOWUP_RULES
        parts.append(_ending(rules))
    written_as_text = sorted(RECOVERABLE_TOOLS & names)
    if written_as_text:
        parts.append(TOOL_CALL_RULE.format(names=", ".join(written_as_text)))
    return "\n".join(parts)
