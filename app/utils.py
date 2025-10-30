"""Shared utility functions for the Iris Assistant."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def timestamp_id() -> str:
    """Return a filesystem-friendly UTC timestamp identifier."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def ensure_directory(path: os.PathLike[str] | str) -> Path:
    """Create the directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def env_flag(name: str, default: bool = False) -> bool:
    """Return an environment flag where truthy strings map to True."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: Optional[str] = None) -> Path:
    """Return an environment variable as a Path, erroring if missing."""
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return Path(value)

