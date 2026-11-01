from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class MemoryRecord:
    """Un dato almacenado en la memoria a largo plazo de un usuario."""

    def __init__(
        self,
        user_id: str,
        key: str,
        value: str,
        timestamp: datetime | None = None,
    ) -> None:
        self.user_id = user_id
        self.key = key
        self.value = value
        self.timestamp = timestamp or datetime.now(timezone.utc)


class InMemoryMemoryService:
    """Servicio de memoria a largo plazo en memoria.

    Almacena pares clave/valor por usuario. Útil para recordar
    preferencias y contexto entre sesiones.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[MemoryRecord]] = {}

    async def add(
        self, user_id: str, key: str, value: str
    ) -> None:
        """Guarda un dato en la memoria del usuario."""
        self._records.setdefault(user_id, []).append(
            MemoryRecord(user_id=user_id, key=key, value=value)
        )

    async def get(
        self, user_id: str, key: str
    ) -> str | None:
        """Recupera un dato por clave (devuelve el más reciente)."""
        records = self._records.get(user_id, [])
        for r in reversed(records):
            if r.key == key:
                return r.value
        return None

    async def search(
        self, user_id: str, query: str
    ) -> list[dict[str, Any]]:
        """Busca datos por coincidencia de texto en clave o valor."""
        q = query.lower()
        results: list[dict[str, Any]] = []
        for r in self._records.get(user_id, []):
            if q in r.key.lower() or q in r.value.lower():
                results.append(
                    {
                        "key": r.key,
                        "value": r.value,
                        "timestamp": r.timestamp.isoformat(),
                    }
                )
        return results
