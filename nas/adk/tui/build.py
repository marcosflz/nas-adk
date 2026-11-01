"""Build a Windows console executable from a Python script or an app directory."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import click


def build_exe(
    path: str | Path, *, name: str | None = None, onefile: bool = True,
) -> Path:
    """Bundle a script, or a directory's app.py, using the current Python environment.

    Only the adjacent assets/ directory is copied as application data. Dependencies
    must already be installed. Output goes to the caller's dist/; build/ holds the
    generated spec and work files. Existing distributions are never overwritten.
    """
    source = Path(path).resolve()
    entry = source / "app.py" if source.is_dir() else source
    if not entry.is_file():
        raise FileNotFoundError(f"Python entry point not found: {entry}")
    if entry.suffix.lower() != ".py":
        raise ValueError("Pass a .py file or a directory containing app.py")
    name = name if name is not None else (source.name if source.is_dir() else source.stem)
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
        f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "123456789¹²³"
    }
    if (
        not name or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        or name.endswith((" ", ".")) or name.split(".", 1)[0].rstrip().upper() in reserved
    ):
        raise ValueError("name must be a valid Windows filename, without a path")
    if sys.platform != "win32":
        raise RuntimeError("Windows executables must be built on Windows")
    if find_spec("PyInstaller") is None:
        raise RuntimeError('Install the build extra: pip install "nas-adk[tui,build]"')

    output = Path.cwd() / "dist"
    executable = name if name.lower().endswith(".exe") else f"{name}.exe"
    result = output / (executable if onefile else name)
    if result.exists():
        raise FileExistsError(f"Output already exists: {result}. Choose another --name.")
    work = Path.cwd() / "build" / name
    command = [
        sys.executable, "-m", "PyInstaller",
        "--onefile" if onefile else "--onedir", "--console", "--name", name,
        "--distpath", str(output), "--workpath", str(work / "work"),
        "--specpath", str(work), "--paths", str(entry.parent),
    ]
    assets = entry.parent / "assets"
    if assets.is_dir():
        command.extend(["--add-data", f"{assets};assets"])
    command.append(str(entry))
    subprocess.run(command, cwd=entry.parent, check=True)
    if not (result if onefile else result / executable).is_file():
        raise RuntimeError(f"PyInstaller did not produce the expected executable: {result}")
    return result


@click.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--name", default=None, help="Executable name; defaults to the file or directory name.")
@click.option("--onefile/--onedir", default=True, show_default=True,
              help="One console executable, or a directory for debugging.")
def main(path: Path, name: str | None, onefile: bool) -> None:
    """Build a Windows TUI executable from a .py file or a directory with app.py."""
    try:
        result = build_exe(path, name=name, onefile=onefile)
    except subprocess.CalledProcessError as error:
        raise click.ClickException(f"PyInstaller failed (exit code {error.returncode})") from error
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Built: {result}")


if __name__ == "__main__":
    main()
