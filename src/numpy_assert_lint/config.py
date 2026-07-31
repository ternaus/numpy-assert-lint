"""Configuration loading for numpy-assert-lint."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

DEFAULT_SELECT = ("NAL001", "NAL002", "NAL003", "NAL004")


class ConfigError(ValueError):
    """Raised when numpy-assert-lint configuration has an invalid shape."""


@dataclass(frozen=True)
class Config:
    """Enabled and ignored rule selectors."""

    select: tuple[str, ...] = DEFAULT_SELECT
    ignore: tuple[str, ...] = ()


def load_config(path: Path) -> Config:
    """Load configuration from a pyproject.toml file when it exists."""
    if not path.is_file():
        return Config()

    try:
        with path.open("rb") as config_file:
            document: dict[str, Any] = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(str(error)) from error
    tool_table = document.get("tool", {})
    if not isinstance(tool_table, dict):
        message = "tool must be a table"
        raise ConfigError(message)
    table = tool_table.get("numpy-assert-lint", {})
    if not isinstance(table, dict):
        message = "tool.numpy-assert-lint must be a table"
        raise ConfigError(message)
    return Config(
        select=_read_selectors(table.get("select"), name="select", default=DEFAULT_SELECT),
        ignore=_read_selectors(table.get("ignore"), name="ignore", default=()),
    )


def _read_selectors(value: object, *, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        message = f"{name} must be an array of strings"
        raise ConfigError(message)
    return tuple(item.upper() for item in value)
