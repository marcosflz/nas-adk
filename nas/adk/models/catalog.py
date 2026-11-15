"""Catálogo de modelos que ofrece un proveedor a la cuenta autenticada."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Un modelo elegible: ``id`` es lo que se pasa como ``model``; ``label``, lo que se muestra."""

    id: str
    label: str
    description: str | None = None
    efforts: list[str] = Field(default_factory=list)
