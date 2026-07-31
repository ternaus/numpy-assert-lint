"""Command-line interface for numpy-assert-lint."""

from __future__ import annotations

import argparse
import difflib
import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from numpy_assert_lint import __version__
from numpy_assert_lint.checker import Diagnostic, check_source
from numpy_assert_lint.config import ConfigError, load_config
from numpy_assert_lint.fixer import FixResult, fix_source

_SKIPPED_DIRECTORIES = {"build", "dist", "node_modules", "site-packages"}


class _Arguments(argparse.Namespace):
    config: Path | None
    select: str | None
    ignore: str | None
    fix: bool
    diff: bool
    unsafe_fixes: bool
    filenames: list[str]


@dataclass(frozen=True)
class _Options:
    selected_selectors: tuple[str, ...]
    ignored_selectors: tuple[str, ...]
    fix: bool
    diff: bool
    unsafe_fixes: bool


@dataclass(frozen=True)
class _CheckedFile:
    source: str
    encoding: str
    diagnostics: list[Diagnostic]


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(prog="numpy-assert-lint")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="Path to a pyproject.toml file.")
    parser.add_argument("--select", help="Comma-separated rule codes or prefixes to enable.")
    parser.add_argument("--ignore", help="Comma-separated rule codes or prefixes to ignore.")
    change_group = parser.add_mutually_exclusive_group()
    change_group.add_argument("--fix", action="store_true", help="Apply safe fixes in place.")
    change_group.add_argument("--diff", action="store_true", help="Print safe fixes as a unified diff.")
    parser.add_argument("--unsafe-fixes", action="store_true", help="Allow fixes that can change comparison semantics.")
    parser.add_argument("filenames", nargs="*", default=["."])
    arguments = _Arguments()
    parser.parse_args(argv, namespace=arguments)
    if arguments.unsafe_fixes and not (arguments.fix or arguments.diff):
        parser.error("--unsafe-fixes requires --fix or --diff")
    return arguments


def _resolve_options(arguments: _Arguments) -> _Options | None:
    config_path = arguments.config or Path("pyproject.toml")
    if arguments.config is not None and not config_path.is_file():
        sys.stderr.write(f"{config_path}: NAL901 Configuration file not found\n")
        return None
    try:
        config = load_config(config_path)
    except ConfigError as error:
        sys.stderr.write(f"{config_path}: NAL901 Invalid configuration: {error}\n")
        return None
    selected_selectors = _parse_selectors(arguments.select) if arguments.select is not None else config.select
    ignored_selectors = _parse_selectors(arguments.ignore) if arguments.ignore is not None else config.ignore
    return _Options(
        selected_selectors=selected_selectors,
        ignored_selectors=ignored_selectors,
        fix=arguments.fix,
        diff=arguments.diff,
        unsafe_fixes=arguments.unsafe_fixes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Lint the requested Python files and return a process exit code."""
    arguments = _parse_arguments(argv)
    options = _resolve_options(arguments)
    if options is None:
        return 2

    python_files, missing_paths = _iter_python_files(arguments.filenames)
    exit_code = _report_missing_paths(missing_paths)
    for path in python_files:
        exit_code = max(exit_code, _process_file(path, options))
    return exit_code


def _report_missing_paths(missing_paths: list[Path]) -> int:
    for path in missing_paths:
        sys.stderr.write(f"{path}: NAL902 File not found\n")
    return 2 if missing_paths else 0


def _process_file(path: Path, options: _Options) -> int:
    checked_file = _check_file(path)
    if checked_file is None:
        return 2

    diagnostics = checked_file.diagnostics
    changed = False
    if options.fix or options.diff:
        diagnostics, changed = _apply_fixes(path, checked_file, options)
    violations_found = _report_diagnostics(diagnostics, options)
    return 1 if changed or violations_found else 0


def _check_file(path: Path) -> _CheckedFile | None:
    try:
        source, encoding = _read_source(path)
    except (OSError, SyntaxError, UnicodeError) as error:
        message = (
            error.msg.split(" for ", maxsplit=1)[0].removesuffix(" declaration")
            if isinstance(error, SyntaxError)
            else str(error)
        )
        sys.stderr.write(f"{path}: NAL903 Could not read file: {message}\n")
        return None

    try:
        diagnostics = check_source(source, filename=str(path))
    except SyntaxError as error:
        line = error.lineno or 1
        column = error.offset or 1
        sys.stderr.write(f"{path}:{line}:{column}: NAL900 SyntaxError: {error.msg}\n")
        return None
    return _CheckedFile(source=source, encoding=encoding, diagnostics=diagnostics)


def _apply_fixes(path: Path, checked_file: _CheckedFile, options: _Options) -> tuple[list[Diagnostic], bool]:
    fix_result = fix_source(
        checked_file.source,
        filename=str(path),
        enabled_codes=_enabled_codes(checked_file.diagnostics, options),
        allow_unsafe=options.unsafe_fixes,
        diagnostics=checked_file.diagnostics,
    )
    if not fix_result.fixed_codes:
        return checked_file.diagnostics, False

    _emit_fix(path, checked_file, fix_result, as_diff=options.diff)
    remaining_diagnostics = [] if options.diff else check_source(fix_result.source, filename=str(path))
    return remaining_diagnostics, True


def _emit_fix(path: Path, checked_file: _CheckedFile, fix_result: FixResult, *, as_diff: bool) -> None:
    if as_diff:
        sys.stdout.writelines(
            difflib.unified_diff(
                checked_file.source.splitlines(keepends=True),
                fix_result.source.splitlines(keepends=True),
                fromfile=f"{path}:before",
                tofile=f"{path}:after",
            )
        )
        return

    path.write_bytes(fix_result.source.encode(checked_file.encoding))
    fixed_count = len(fix_result.fixed_codes)
    noun = "violation" if fixed_count == 1 else "violations"
    sys.stderr.write(f"{path}: Fixed {fixed_count} {noun}.\n")


def _enabled_codes(diagnostics: list[Diagnostic], options: _Options) -> set[str]:
    return {diagnostic.code for diagnostic in diagnostics if _is_enabled(diagnostic.code, options)}


def _report_diagnostics(diagnostics: list[Diagnostic], options: _Options) -> bool:
    violations_found = False
    for diagnostic in diagnostics:
        if not _is_enabled(diagnostic.code, options):
            continue
        violations_found = True
        sys.stdout.write(
            f"{diagnostic.filename}:{diagnostic.line}:{diagnostic.column}: {diagnostic.code} {diagnostic.message}\n"
        )
    return violations_found


def _is_enabled(code: str, options: _Options) -> bool:
    return _matches_selector(code, options.selected_selectors) and not _matches_selector(
        code,
        options.ignored_selectors,
    )


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


def _read_source(path: Path) -> tuple[str, str]:
    source_bytes = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
    return source_bytes.decode(encoding), encoding
