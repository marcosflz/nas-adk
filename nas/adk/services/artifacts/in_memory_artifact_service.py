from __future__ import annotations

from datetime import datetime, timezone


class Artifact:
    """Archivo o blob almacenado en una sesión."""

    def __init__(
        self,
        id: str,
        session_id: str,
        name: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> None:
        self.id = id
        self.session_id = session_id
        self.name = name
        self.content = content
        self.mime_type = mime_type
        self.created_at = datetime.now(timezone.utc)


class InMemoryArtifactService:
    """Servicio de artefactos en memoria.

    Permite guardar y recuperar archivos asociados a una sesión.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    async def save(
        self,
        session_id: str,
        name: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> Artifact:
        """Guarda un archivo en la sesión."""
        import uuid

        art = Artifact(
            id=str(uuid.uuid4()),
            session_id=session_id,
            name=name,
            content=content,
            mime_type=mime_type,
        )
        self._artifacts[art.id] = art
        return art

    async def get(self, artifact_id: str) -> Artifact | None:
        """Recupera un artefacto por ID."""
        return self._artifacts.get(artifact_id)

    async def list_by_session(
        self, session_id: str
    ) -> list[Artifact]:
        """Lista todos los artefactos de una sesión."""
        return [
            a for a in self._artifacts.values() if a.session_id == session_id
        ]
