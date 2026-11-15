"""Catálogo de modelos que ofrece un proveedor a la cuenta autenticada."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Un modelo elegible: ``id`` es lo que se pasa como ``model``; ``label``, lo que se muestra."""

    id: str
    label: str
    description: str | None = None
    efforts: list[str] = Field(default_factory=list)
    source: Literal["account", "known"] = "account"
    """``account``: lo anuncia la cuenta autenticada. ``known``: está en el catálogo conocido del
    proveedor, pero la cuenta no lo anunció (puede no estar en su plan)."""
