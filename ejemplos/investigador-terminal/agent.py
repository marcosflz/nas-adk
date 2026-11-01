"""
Investigador de terminal: chat LLM y pipeline determinista de datos ficticios.
El modelo conversa; las herramientas recolectan, analizan, validan y formatean.
Ejecutar: python ejemplos/investigador-terminal/agent.py [--offline].
"""

import argparse
import asyncio
import os
import unicodedata

from nas.adk import Agent, Context, Workflow, node
from nas.adk.cli import run_chat
from nas.adk.models import ModelRouter


# Configuración local: basta editar estas variables; no hace falta un .env.
MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")
OPENCODE_API = os.getenv("OPENCODE_API", "auto")  # auto, chat_completions o responses.
GENERATION_CONFIG: dict[str, object] = {}  # Añade reasoning_effort solo si el modelo lo admite.
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go")
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_FALLBACK_MODELS = os.getenv("OPENCODE_FALLBACK_MODELS", "")
REQUEST_TIMEOUT = 120.0
OFFLINE = False  # True para probar el pipeline sin LLM ni conexiones de red.


# ── Mock tools con descripciones auto-explicativas ────────────
# La descripción de cada tool es lo que el LLM lee para decidir
# si usarla. Por eso listamos EXPLÍCITAMENTE los temas cubiertos.
# Si el LLM ve que su tema no está en la lista, sabe que no debe
# llamar la tool, o que si la llama obtendrá NO_ENCONTRADO.


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


_TEMAS = [
    (
        "clima",
        ("climatico", "clima", "calentamiento", "temperatura", "co2"),
        "NAS 2025: +2.31°C, CO2 512ppm, mar +8.7mm/año. Permafrost libera 2.3Gt metano extra.",
    ),
    (
        "solar",
        ("solar", "panel", "fotovoltaico"),
        "NAS 2025: 3.2TW instalados, +42% anual. Granjas flotantes Mar del Norte: 180TWh.",
    ),
    (
        "vehiculos_electricos",
        (
            "coche electrico",
            "vehiculo electrico",
            "electricos",
            "coche",
            "vehiculo",
            "ev",
            "byd",
            "tesla",
            "bateria",
            "hibrido",
            "mercado",
        ),
        "NAS 2025: 8.2M ventas globales (+18%), cuota 14%. "
        "BYD 4.1M, Tesla 1.8M, VW 0.9M. China 52% del mercado. "
        "Baterías estado sólido: 50GWh/año. Precio medio: $42K.",
    ),
    (
        "cripto",
        ("cripto", "bitcoin", "ethereum", "btc", "eth", "blockchain"),
        "NAS 2025: BTC $48.2K, ETH $3.15K, cap $1.2T. ETFs BTC: $85B AUM. Consumo -35% post-halving.",
    ),
]

_TEMAS_ESTADISTICAS = [
    (
        "clima",
        ("climatico", "clima", "calentamiento", "temperatura"),
        {2024: "41.2Gt CO2e, +2.11°C", 2025: "43.8Gt, +2.31°C"},
    ),
    ("solar", ("solar", "panel", "fotovoltaico"), {2024: "2.1TW, $620B", 2025: "3.2TW, $890B"}),
    (
        "vehiculos_electricos",
        (
            "coche electrico",
            "vehiculo electrico",
            "electricos",
            "coche",
            "vehiculo",
            "ev",
            "byd",
            "tesla",
            "bateria",
            "hibrido",
            "mercado",
        ),
        {2024: "8.2M uds, 450GWh", 2025: "11.5M, 680GWh"},
    ),
    (
        "cripto",
        ("cripto", "bitcoin", "ethereum", "btc", "eth"),
        {2024: "BTC $42K, ETH $2.8K", 2025: "BTC $48.2K, ETH $3.15K"},
    ),
]


def buscar_datos(tema: str) -> str:
    """Busca datos del observatorio NAS.

    Temas disponibles: cambio climático, energía solar,
    vehículos eléctricos, criptomonedas.
    Si el tema no está cubierto, devuelve NO_ENCONTRADO.
    """
    tema_norm = _normalize(tema)
    for name, keywords, data in _TEMAS:
        if _match(tema_norm, keywords):
            return data
    return "NO_ENCONTRADO"


def obtener_estadisticas(tema: str, año: int = 2025) -> str:
    """Estadísticas numéricas NAS para un tema y año.
    Si no hay datos, devuelve NO_ENCONTRADO.
    """
    tema_norm = _normalize(tema)
    for name, keywords, años in _TEMAS_ESTADISTICAS:
        if _match(tema_norm, keywords):
            return años.get(año, "NO_ENCONTRADO")
    return "NO_ENCONTRADO"


def _match(tema_norm: str, keywords: tuple[str, ...]) -> bool:
    for kw in keywords:
        if kw in tema_norm:
            idx = tema_norm.find(kw)
            before = idx == 0 or not tema_norm[idx - 1].isalnum()
            end = idx + len(kw)
            after = (
                end >= len(tema_norm)
                or not tema_norm[end].isalnum()
                or tema_norm[end] in "sn"
                and (end + 1 >= len(tema_norm) or not tema_norm[end + 1].isalnum())
            )
            if before and after:
                return True
    return False


def analizar_tendencia(datos: str, periodo: str = "5a") -> str:
    """Analiza tendencia a partir de datos NAS.

    Detecta patrones específicos en los datos:
    - Si los datos mencionan temperaturas (2.31°C) → tendencia climática
    - Si mencionan capacidad solar (3.2TW) → tendencia energética
    - Si mencionan ventas (8.2M) → tendencia mercado VE
    - En otro caso → tendencia genérica
    """
    if "2.31" in datos:
        return "TENDENCIA: calentamiento +0.86°C sobre IPCC. Metano factor dominante. +3.1°C proyección 2030."
    if "3.2TW" in datos:
        return (
            "TENDENCIA: solar +42% anual. Granjas flotantes 60% new capacity. 5TW proyección 2027."
        )
    if "8.2M" in datos:
        return "TENDENCIA: VE +18% anual. Baterías estado sólido cuello de botella. 15M proyección 2026."
    return "TENDENCIA: crecimiento moderado."


def calcular_metricas(datos: str) -> str:
    """Calcula métricas derivadas de datos NAS."""
    return "MÉTRICAS: +22-35% interanual, eficiencia +18%, retorno 3.2x/5a, TRL 7-9"


def verificar_fuente(fuente: str, dato: str) -> str:
    """Verifica que los datos provengan de fuente NAS (nuestro observatorio)."""
    if "NAS" in fuente or "Observatorio" in fuente:
        return "✅ Fuente NAS: datos verificados"
    return "❌ No es fuente NAS"


def chequear_coherencia(informe: str) -> str:
    """Revisa que el informe use datos NAS, métricas y tendencias."""
    ok = (
        "NAS" in informe
        and any(s in informe for s in ["%", "$", "TW", "Gt"])
        and "TENDENCIA" in informe
    )
    if ok:
        return "✅ Coherente y completo con datos NAS"
    return "❌ Incompleto. Recolecta datos NAS específicos."


def formatear_informe(contenido: str, formato: str = "markdown") -> str:
    """Da formato final al informe NAS."""
    return (
        f"=== INFORME NAS ===\n"
        f"Fuente: Observatorio NAS 2025 | Formato: {formato}\n\n"
        f"{contenido}\n\n=== FIN ==="
    )


def preguntar_al_usuario(pregunta: str) -> str:
    """Pregunta al usuario solo si las otras herramientas devuelven NO_ENCONTRADO."""
    return f"AYUDA: {pregunta}"


# La secuencia es fija y las tools son deterministas: no necesita inferencia.
@node
def recolectar(ctx: Context, tema: str) -> dict:
    datos = buscar_datos(tema)
    return {
        "tema": tema,
        "datos": datos,
        "estadisticas": obtener_estadisticas(tema) if datos != "NO_ENCONTRADO" else "",
    }


@node
def analizar(ctx: Context, informe: dict) -> dict:
    if informe["datos"] != "NO_ENCONTRADO":
        informe["analisis"] = analizar_tendencia(informe["datos"])
        informe["metricas"] = calcular_metricas(informe["datos"])
    return informe


@node
async def validar(ctx: Context, informe: dict) -> dict:
    if informe["datos"] == "NO_ENCONTRADO":
        return informe
    for attempt in range(2):
        contenido = "\n".join(str(value) for value in informe.values())
        fuente = verificar_fuente("NAS", informe["datos"])
        coherencia = chequear_coherencia(contenido)
        if fuente.startswith("✅") and coherencia.startswith("✅"):
            informe["validacion"] = f"{fuente}\n{coherencia}"
            return informe
        if attempt == 0:
            informe = await analizar.run(ctx, await recolectar.run(ctx, informe["tema"]))
    raise ValueError("La validación NAS falló tras dos intentos")


@node
def redactar(ctx: Context, informe: dict) -> str:
    if informe["datos"] == "NO_ENCONTRADO":
        return preguntar_al_usuario(
            "No hay datos NAS para ese tema. ¿Qué tema disponible quieres investigar?"
        )
    return formatear_informe("\n".join(str(value) for value in informe.values()))


offline_agent = Workflow(
    name="investigador",
    edges=[
        ("START", recolectar),
        (recolectar, analizar),
        (analizar, validar),
        (validar, redactar),
    ],
)


async def investigar(tema: str, ctx: Context) -> str:
    """Genera un informe con datos FICTICIOS de demostración.

    Temas disponibles: cambio climático, energía solar, vehículos eléctricos,
    criptomonedas. Ejecuta recolección, análisis, validación y redacción.
    No consulta fuentes reales ni internet. No usar como información factual.
    """
    return await ctx.run_agent(offline_agent, tema)


root_agent = Agent(
    name="investigador",
    model=ModelRouter(
        provider="opencode",
        base_url=OPENCODE_BASE_URL,
        api_key=OPENCODE_API_KEY,
        model=MODEL,
        api=OPENCODE_API if OPENCODE_API in {"auto", "chat_completions", "responses"} else "auto",
        fallback_models=tuple(
            model.strip() for model in OPENCODE_FALLBACK_MODELS.split(",") if model.strip()
        ),
        timeout=REQUEST_TIMEOUT,
    ),
    generation_config=GENERATION_CONFIG,
    instruction=(
        "Eres un investigador conversacional en español. Recuerda el contexto del chat. "
        "Para obtener datos o informes utiliza investigar con un tema concreto; "
        "la herramienta ya ejecuta la secuencia recolectar, analizar, validar y redactar. "
        "Sus datos son FICTICIOS: indícalo siempre al presentarlos. No inventes cifras ni fuentes. "
        "Puedes explicar o resumir informes anteriores sin repetir la investigación. "
        "Si no hay datos, pide al usuario otro tema. Responde naturalmente y permite preguntas de seguimiento."
    ),
    tools=[investigar],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Datos ficticios, sin LLM ni red")
    mode.add_argument("--model", help="Modelo OpenCode; por defecto usa MODEL de este fichero")
    args = parser.parse_args()
    offline = args.offline or (OFFLINE and args.model is None)
    model_name = args.model or MODEL
    base_url = OPENCODE_BASE_URL or os.getenv("OPENCODE_BASE_URL", "")
    api_key = OPENCODE_API_KEY or os.getenv("OPENCODE_API_KEY", "")
    if not offline:
        if OPENCODE_API not in {"auto", "chat_completions", "responses"}:
            parser.error("OPENCODE_API debe ser 'auto', 'chat_completions' o 'responses'.")
        if not base_url or not api_key.strip():
            parser.error("Configura OPENCODE_BASE_URL y OPENCODE_API_KEY o usa --offline.")

    async def execute() -> None:
        if offline:
            print("Modo offline: introduce un tema; sin inferencia ni llamadas de red.")
            await run_chat(offline_agent)
        else:
            print(f"Modelo OpenCode: {model_name}. Las llamadas al proveedor pueden consumir tokens.")
            model = ModelRouter(
                provider="opencode", base_url=base_url, api_key=api_key, model=model_name,
                api=OPENCODE_API,
                fallback_models=tuple(
                    model.strip() for model in OPENCODE_FALLBACK_MODELS.split(",") if model.strip()
                ),
                timeout=REQUEST_TIMEOUT,
            )
            root_agent.model = model
            root_agent.generation_config = GENERATION_CONFIG
            try:
                await run_chat(root_agent)
            finally:
                await model.aclose()

    print("Datos FICTICIOS. Temas: clima, energia solar, vehiculos electricos, criptomonedas.")
    try:
        asyncio.run(execute())
    except KeyboardInterrupt:
        print("\nChat finalizado.")


if __name__ == "__main__":
    main()
