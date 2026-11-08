"""Función Python compartida por los dos ejemplos (datos ficticios y deterministas)."""

from __future__ import annotations

import json

GASTOS = {"2026-10": 1240.50, "2026-11": 987.25}


def consultar_gastos(mes: str) -> str:
    """Devuelve el total de gastos de un mes con formato YYYY-MM."""
    if mes not in GASTOS:
        return json.dumps({"error": f"sin datos para {mes}"})
    return json.dumps({"mes": mes, "total_eur": GASTOS[mes]})
