from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_executable(
    name: str,
    configured: str | None,
    project_root: Path,
) -> Path:
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and os.access(configured_path, os.X_OK):
            return configured_path.resolve()
        resolved = shutil.which(configured)
        if resolved:
            return Path(resolved)
        raise FileNotFoundError(f"Configured {name} executable is unavailable: {configured}")

    resolved = shutil.which(name)
    if resolved:
        return Path(resolved)
    local_executable = project_root / ".venv" / "bin" / name
    if local_executable.is_file() and os.access(local_executable, os.X_OK):
        return local_executable
    raise FileNotFoundError(
        f"{name} was not found on PATH or at {local_executable}."
    )
