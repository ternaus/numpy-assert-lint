"""AST checks for NumPy assertions."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

_DIRECT_CALL_RULES = {
    "numpy.allclose": ("NAL001", "Prefer np.testing.assert_allclose() for diagnostic output."),
    "numpy.array_equal": ("NAL002", "Prefer np.testing.assert_array_equal() for diagnostic output."),
}
_NOQA_PATTERN = re.compile(r"#\s*noqa(?:\s*:\s*([A-Z0-9_,\s]+))?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Diagnostic:
    """A lint violation tied to a source location."""

    filename: str
    line: int
    column: int
    code: str
    message: str


class _AssertionVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, noqa_by_line: dict[int, frozenset[str] | None]) -> None:
        self.filename = filename
        self.noqa_by_line = noqa_by_line
        self.aliases: dict[str, str] = {}
        self.diagnostics: list[Diagnostic] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            binding = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if alias.name == "numpy" or (alias.name.startswith("numpy.") and alias.asname is None):
                self.aliases[binding] = "numpy"
            elif alias.name.startswith("numpy."):
                self.aliases[binding] = alias.name
            else:
                self.aliases.pop(binding, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.aliases.clear()
                continue
            binding = alias.asname or alias.name
            if node.level == 0 and node.module == "numpy":
                self.aliases[binding] = f"numpy.{alias.name}"
            else:
                self.aliases.pop(binding, None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.aliases.pop(node.name, None)
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.aliases.pop(node.name, None)
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        outer_aliases = self.aliases
        self.aliases = outer_aliases.copy()
        for statement in node.body:
            self.visit(statement)
        self.aliases = outer_aliases
        self.aliases.pop(node.name, None)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.aliases.pop(node.id, None)

    def visit_Assert(self, node: ast.Assert) -> None:
        rule = self._rule_for_assert(node.test)
        if rule is not None:
            code, message = rule
            if not self._is_suppressed(node, code):
                self.diagnostics.append(
                    Diagnostic(
                        filename=self.filename,
                        line=node.lineno,
                        column=node.col_offset + 1,
                        code=code,
                        message=message,
                    )
                )
        self.generic_visit(node)

    def _is_suppressed(self, node: ast.Assert, code: str) -> bool:
        end_line = node.end_lineno or node.lineno
        for line_number in range(node.lineno, end_line + 1):
            if line_number not in self.noqa_by_line:
                continue
            selected_codes = self.noqa_by_line[line_number]
            if selected_codes is None or code in selected_codes:
                return True
        return False

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        outer_aliases = self.aliases
        self.aliases = outer_aliases.copy()
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            self.aliases.pop(argument.arg, None)

        for statement in node.body:
            self.visit(statement)
        self.aliases = outer_aliases

    def _rule_for_assert(self, test: ast.expr) -> tuple[str, str] | None:
        if not isinstance(test, ast.Call):
            return None

        if (
            not test.args
            and not test.keywords
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "all"
            and self._is_simple_equality(test.func.value)
        ):
            return "NAL005", "Possible array comparison; prefer np.testing.assert_array_equal() for diagnostic output."

        qualified_name = self._qualified_name(test.func)
        direct_rule = _DIRECT_CALL_RULES.get(qualified_name) if qualified_name is not None else None
        if direct_rule is not None:
            return direct_rule

        if qualified_name == "numpy.all" and len(test.args) == 1 and not test.keywords:
            operand = test.args[0]
            if isinstance(operand, ast.Call) and self._qualified_name(operand.func) == "numpy.isclose":
                return "NAL003", "Prefer np.testing.assert_allclose() for diagnostic output."
            if self._is_simple_equality(operand):
                return "NAL004", "Prefer np.testing.assert_array_equal() for diagnostic output."

        return None

    @staticmethod
    def _is_simple_equality(node: ast.expr) -> bool:
        return isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)

    def _qualified_name(self, node: ast.expr) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if not isinstance(node, ast.Name):
            return None

        root = self.aliases.get(node.id)
        if root is None:
            return None

        return ".".join([root, *reversed(parts)])


def check_source(source: str, *, filename: str = "<unknown>") -> list[Diagnostic]:
    """Return NumPy assertion diagnostics for one Python source string."""
    tree = ast.parse(source, filename=filename)
    visitor = _AssertionVisitor(filename, _collect_noqa(source))
    visitor.visit(tree)
    return visitor.diagnostics


def _collect_noqa(source: str) -> dict[int, frozenset[str] | None]:
    suppressions: dict[int, frozenset[str] | None] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = _NOQA_PATTERN.fullmatch(token.string)
        if match is None:
            continue
        raw_codes = match.group(1)
        if raw_codes is None:
            suppressions[token.start[0]] = None
        else:
            suppressions[token.start[0]] = frozenset(code.upper() for code in re.split(r"[,\s]+", raw_codes) if code)
    return suppressions
