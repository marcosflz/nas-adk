"""Herramientas locales. Todas las rutas de archivos son relativas al workspace."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Iterator

from nas.adk.services.code_executor import CodeExecutor
from nas.adk.tools import FunctionTool

MAX_FILE_BYTES = 256 * 1024
MAX_OUTPUT_CHARS = 12000
MAX_SCAN_ENTRIES = 10000
EXCLUDED_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "dist", "build",
    ".ssh", ".aws", ".azure", ".netrc", ".npmrc", ".pypirc",
    "credentials.json", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
}
SECRET_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".jks"}
_is_reserved = getattr(os.path, "isreserved", lambda path: PureWindowsPath(path).is_reserved())


def _excluded(name: str) -> bool:
    name = name.casefold()
    return (
        name in EXCLUDED_NAMES
        or name.startswith(".env")
        or Path(name).suffix in SECRET_SUFFIXES
    )


def resolve_path(workspace: Path, path: str) -> Path:
    """Rechaza rutas absolutas, escapes, archivos sensibles y streams NTFS."""
    requested = Path(path)
    if not path.strip() or requested.is_absolute() or requested.drive or ":" in path:
        raise PermissionError("Usa una ruta relativa al workspace, sin ':'")
    if os.name == "nt" and any(_is_reserved(part) for part in requested.parts):
        raise PermissionError("No se permiten nombres de dispositivos Windows")
    target = (workspace / requested).resolve()
    try:
        relative = target.relative_to(workspace)
    except ValueError as error:
        raise PermissionError("La ruta sale del workspace") from error
    if any(_excluded(part) for part in (*requested.parts, *relative.parts)):
        raise PermissionError("Ruta excluida: metadatos, dependencias o credenciales")
    return target


def _read_text(path: Path) -> str:
    with path.open("rb") as source:
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES or b"\x00" in data:
        raise ValueError("Solo archivos de texto de hasta 256 KiB; no binarios")
    return data.decode("utf-8")


def _check_content(content: str) -> None:
    if "\x00" in content or len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("Solo texto UTF-8 de hasta 256 KiB, sin bytes nulos")


def _bounded(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n[Salida truncada; reduce el rango o la búsqueda]"
    return text


def _files(workspace: Path, start: Path) -> Iterator[Path]:
    if start.is_file():
        yield start
        return
    if not start.is_dir():
        raise FileNotFoundError("No existe el archivo o directorio")
    scanned = 0
    # ponytail: recorrido lineal de 10000 entradas; usar un índice para proyectos mayores.
    for folder, dirs, files in os.walk(start, followlinks=False):
        scanned += len(dirs) + len(files)
        if scanned > MAX_SCAN_ENTRIES:
            raise ValueError("Demasiadas entradas; busca dentro de una subcarpeta")
        allowed_dirs = []
        for name in sorted(dirs):
            entry = Path(folder) / name
            if _excluded(name) or entry.is_symlink():
                continue
            try:
                resolve_path(workspace, entry.relative_to(workspace).as_posix())
            except (PermissionError, ValueError, OSError):
                continue
            allowed_dirs.append(name)
        dirs[:] = allowed_dirs
        for name in sorted(files):
            entry = Path(folder) / name
            if _excluded(name) or entry.is_symlink():
                continue
            yield resolve_path(workspace, entry.relative_to(workspace).as_posix())


def build_tools(workspace: Path, allow_commands: bool = False) -> list[FunctionTool]:
    """Prepara funciones que comparten una raíz validada; no modifica el cwd."""
    workspace = workspace.expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("El workspace debe ser un directorio existente")

    def list_files(path: str = ".", offset: int = 0, limit: int = 100) -> str:
        """Lista archivos recursivamente; pagina con offset. Omite rutas excluidas y enlaces."""
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("offset >= 0 y limit entre 1 y 200")
        results: list[str] = []
        for index, file in enumerate(_files(workspace, resolve_path(workspace, path))):
            if index < offset:
                continue
            if len(results) == limit:
                results.append(f"[Más archivos; usa offset={offset + limit}]")
                break
            results.append(file.relative_to(workspace).as_posix())
        return _bounded("\n".join(results) or "Sin archivos")

    def read_file(path: str, start_line: int = 1, limit: int = 100) -> str:
        """Lee hasta 200 líneas numeradas de un archivo UTF-8 de hasta 256 KiB."""
        if start_line < 1 or not 1 <= limit <= 200:
            raise ValueError("start_line >= 1 y limit entre 1 y 200")
        content = _read_text(resolve_path(workspace, path))
        lines = content.splitlines()
        end = start_line - 1 + limit
        result = "\n".join(
            f"{number}: {line}"
            for number, line in enumerate(lines[start_line - 1:end], start_line)
        )
        if end < len(lines):
            result += f"\n[Más líneas; usa start_line={end + 1}]"
        return _bounded(result or "Sin líneas en este rango")

    def search_text(text: str, path: str = ".", limit: int = 50) -> str:
        """Busca texto literal por línea; omite binarios, no UTF-8 y archivos >256 KiB."""
        if not text or "\n" in text or "\r" in text or not 1 <= limit <= 200:
            raise ValueError("Texto no vacío de una línea y limit entre 1 y 200")
        results: list[str] = []
        for file in _files(workspace, resolve_path(workspace, path)):
            try:
                content = _read_text(file)
            except (OSError, UnicodeError, ValueError):
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if text in line:
                    if len(results) == limit:
                        return _bounded("\n".join(results) + "\n[Más coincidencias; acota la búsqueda]")
                    results.append(f"{file.relative_to(workspace).as_posix()}:{number}: {line}")
        return _bounded("\n".join(results) or "Sin coincidencias en los archivos examinados")

    def create_file(path: str, content: str) -> str:
        """Crea un archivo UTF-8 y sus directorios. Nunca sobrescribe un archivo existente."""
        target = resolve_path(workspace, path)
        _check_content(content)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="") as destination:
            destination.write(content)
        return f"Creado: {path}"

    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Reemplaza texto exacto una sola vez, sin números de línea. Conserva CRLF/LF."""
        target = resolve_path(workspace, path)
        original = _read_text(target)
        position = original.find(old_text)
        if not old_text or position < 0 or original.find(old_text, position + 1) >= 0:
            raise ValueError("old_text debe aparecer exactamente una vez y no estar vacío")
        replacement = original.replace(old_text, new_text, 1)
        _check_content(replacement)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as destination:
                temporary = Path(destination.name)
                destination.write(replacement)
            temporary.chmod(target.stat().st_mode)
            if _read_text(target) != original:
                raise ValueError("El archivo cambió durante la edición; vuelve a leerlo")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return f"Editado: {path}"

    async def run_command(command: str, timeout: float = 30) -> dict[str, str | int]:
        """Ejecuta shell solo con ALLOW_COMMANDS=True. No es un sandbox; conserva permisos y entorno."""
        if not allow_commands:
            raise PermissionError("Comandos deshabilitados; requieren ALLOW_COMMANDS=True en agent.py")
        if not command.strip() or "\x00" in command or len(command) > 4096 or not 0 < timeout <= 30:
            raise ValueError("Comando no vacío de hasta 4096 caracteres; timeout entre 0 y 30s")
        # CodeExecutor hereda el cwd; el launcher lo fija una vez, nunca desde una tool.
        if Path.cwd().resolve() != workspace:
            raise RuntimeError("El directorio de trabajo no coincide con el workspace")
        result = await CodeExecutor(
            trusted=True, timeout=timeout, max_output_bytes=4096,
        ).run_shell(command)
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}

    # Sin parallel_safe: también los hijos operan secuencialmente sobre el mismo workspace.
    return [FunctionTool(fn) for fn in (
        list_files, read_file, search_text, create_file, edit_file, run_command,
    )]
