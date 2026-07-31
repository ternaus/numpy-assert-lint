"""Command-line interface for numpy-assert-lint."""

from __future__ import annotations

import argparse
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

from numpy_assert_lint import __version__
from numpy_assert_lint.checker import check_source
from numpy_assert_lint.config import ConfigError, load_config

_SKIPPED_DIRECTORIES = {"build", "dist", "node_modules", "site-packages"}


def main(argv: Sequence[str] | None = None) -> int:
    """Lint the requested Python files and return a process exit code."""
    parser = argparse.ArgumentParser(prog="numpy-assert-lint")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="Path to a pyproject.toml file.")
    parser.add_argument("--select", help="Comma-separated rule codes or prefixes to enable.")
    parser.add_argument("--ignore", help="Comma-separated rule codes or prefixes to ignore.")
    parser.add_argument("filenames", nargs="*", default=["."])
    arguments = parser.parse_args(argv)
    config_path = arguments.config or Path("pyproject.toml")
    if arguments.config is not None and not config_path.is_file():
        sys.stderr.write(f"{config_path}: NAL901 Configuration file not found\n")
        return 2
    try:
        config = load_config(config_path)
    except ConfigError as error:
        sys.stderr.write(f"{config_path}: NAL901 Invalid configuration: {error}\n")
        return 2
    selected_selectors = _parse_selectors(arguments.select) if arguments.select is not None else config.select
    ignored_selectors = _parse_selectors(arguments.ignore) if arguments.ignore is not None else config.ignore

    python_files, missing_paths = _iter_python_files(arguments.filenames)
    exit_code = 0
    for path in missing_paths:
        sys.stderr.write(f"{path}: NAL902 File not found\n")
        exit_code = 2

    for path in python_files:
        try:
            with tokenize.open(path) as source_file:
                source = source_file.read()
        except (OSError, SyntaxError, UnicodeError) as error:
            message = (
                error.msg.split(" for ", maxsplit=1)[0].removesuffix(" declaration")
                if isinstance(error, SyntaxError)
                else str(error)
            )
            sys.stderr.write(f"{path}: NAL903 Could not read file: {message}\n")
            exit_code = 2
            continue
        try:
            diagnostics = check_source(source, filename=str(path))
        except SyntaxError as error:
            line = error.lineno or 1
            column = error.offset or 1
            sys.stderr.write(f"{path}:{line}:{column}: NAL900 SyntaxError: {error.msg}\n")
            exit_code = 2
            continue

        for diagnostic in diagnostics:
            if not _matches_selector(diagnostic.code, selected_selectors) or _matches_selector(
                diagnostic.code, ignored_selectors
            ):
                continue
            exit_code = max(exit_code, 1)
            sys.stdout.write(
                f"{diagnostic.filename}:{diagnostic.line}:{diagnostic.column}: {diagnostic.code} {diagnostic.message}\n"
            )

    return exit_code


def _parse_selectors(raw_selectors: str) -> tuple[str, ...]:
    return tuple(selector.strip().upper() for selector in raw_selectors.split(",") if selector.strip())


def _matches_selector(code: str, selectors: Sequence[str]) -> bool:
    return any(code.startswith(selector) for selector in selectors)


def _iter_python_files(raw_paths: Sequence[str]) -> tuple[list[Path], list[Path]]:
    files: set[Path] = set()
    missing_paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_file():
            if path.suffix == ".py":
                files.add(path)
            continue
        if not path.is_dir():
            missing_paths.append(path)
            continue
        for candidate in path.rglob("*.py"):
            relative_parts = candidate.relative_to(path).parts[:-1]
            if any(part.startswith(".") or part in _SKIPPED_DIRECTORIES for part in relative_parts):
                continue
            files.add(candidate)
    return sorted(files), missing_paths
