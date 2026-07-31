"""Concrete-syntax-tree fixes for NumPy assertions."""

from __future__ import annotations

import ast
from collections.abc import Collection
from dataclasses import dataclass

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from numpy_assert_lint.checker import Diagnostic, check_source

_MAX_LINE_LENGTH = 120


@dataclass(frozen=True)
class FixResult:
    """Source text and a summary of applied or skipped fixes."""

    source: str
    fixed_codes: tuple[str, ...]
    skipped_unsafe_codes: tuple[str, ...]


class _NumpyAliasVisitor(ast.NodeVisitor):
    """Record an in-scope NumPy module alias for each assertion."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.by_assertion: dict[tuple[int, int], str] = {}
        self.function_outer_aliases: dict[str, str] | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            binding = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if alias.name == "numpy" or (alias.name.startswith("numpy.") and alias.asname is None):
                self.aliases[binding] = "numpy"
            else:
                self.aliases.pop(binding, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.aliases.clear()
            else:
                self.aliases.pop(alias.asname or alias.name, None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.aliases.pop(node.name, None)
        visible_aliases = self.aliases if self.function_outer_aliases is None else self.function_outer_aliases
        self._visit_function(node, visible_aliases=visible_aliases)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.aliases.pop(node.name, None)
        visible_aliases = self.aliases if self.function_outer_aliases is None else self.function_outer_aliases
        self._visit_function(node, visible_aliases=visible_aliases)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        outer_aliases = self.aliases
        outer_function_aliases = self.function_outer_aliases
        self.aliases = outer_aliases.copy()
        self.function_outer_aliases = outer_aliases
        for statement in node.body:
            self.visit(statement)
        self.aliases = outer_aliases
        self.function_outer_aliases = outer_function_aliases
        self.aliases.pop(node.name, None)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.aliases.pop(node.id, None)

    def visit_Assert(self, node: ast.Assert) -> None:
        module_aliases = [name for name, qualified_name in self.aliases.items() if qualified_name == "numpy"]
        if module_aliases:
            preferred_alias = next(
                (name for name in ("np", "numpy") if name in module_aliases),
                module_aliases[0],
            )
            self.by_assertion[(node.lineno, node.col_offset + 1)] = preferred_alias
        self.generic_visit(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        visible_aliases: dict[str, str],
    ) -> None:
        outer_aliases = self.aliases
        outer_function_aliases = self.function_outer_aliases
        self.aliases = visible_aliases.copy()
        self.function_outer_aliases = None
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
        self.function_outer_aliases = outer_function_aliases


class _FixTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        diagnostics: list[Diagnostic],
        *,
        allow_unsafe: bool,
        numpy_aliases: dict[tuple[int, int], str],
    ) -> None:
        self.diagnostics = {(item.line, item.column): item for item in diagnostics}
        self.allow_unsafe = allow_unsafe
        self.numpy_aliases = numpy_aliases
        self.fixed_codes: list[str] = []
        self.skipped_unsafe_codes: list[str] = []

    def leave_Assert(self, original_node: cst.Assert, updated_node: cst.Assert) -> cst.BaseSmallStatement:
        position = self.get_metadata(PositionProvider, original_node).start
        diagnostic = self.diagnostics.get((position.line, position.column + 1))
        if diagnostic is None:
            return updated_node

        replacement = _fix_assertion(
            updated_node,
            diagnostic.code,
            allow_unsafe=self.allow_unsafe,
            numpy_alias=self.numpy_aliases.get((position.line, position.column + 1)),
        )
        if replacement is None:
            if (
                not self.allow_unsafe
                and _fix_assertion(
                    updated_node,
                    diagnostic.code,
                    allow_unsafe=True,
                    numpy_alias=self.numpy_aliases.get((position.line, position.column + 1)),
                )
                is not None
            ):
                self.skipped_unsafe_codes.append(diagnostic.code)
            return updated_node

        replacement = _wrap_long_call(replacement, starting_column=position.column)
        self.fixed_codes.append(diagnostic.code)
        return replacement


def fix_source(
    source: str,
    *,
    filename: str = "<unknown>",
    allow_unsafe: bool = False,
    enabled_codes: Collection[str] | None = None,
) -> FixResult:
    """Apply selected NumPy assertion fixes to source text."""
    diagnostics = check_source(source, filename=filename)
    if enabled_codes is not None:
        diagnostics = [item for item in diagnostics if item.code in enabled_codes]
    alias_visitor = _NumpyAliasVisitor()
    alias_visitor.visit(ast.parse(source, filename=filename))
    wrapper = MetadataWrapper(cst.parse_module(source))
    transformer = _FixTransformer(
        diagnostics,
        allow_unsafe=allow_unsafe,
        numpy_aliases=alias_visitor.by_assertion,
    )
    transformed = wrapper.visit(transformer)
    return FixResult(
        source=transformed.code,
        fixed_codes=tuple(transformer.fixed_codes),
        skipped_unsafe_codes=tuple(transformer.skipped_unsafe_codes),
    )


def _fix_numpy_all_equality(node: cst.Assert, *, allow_unsafe: bool) -> cst.Expr | None:
    if not isinstance(node.test, cst.Call):
        return None  # pragma: no cover - guaranteed by the matching diagnostic
    if node.msg is not None and not allow_unsafe and not isinstance(node.msg, cst.SimpleString):
        return None
    call = node.test
    if len(call.args) != 1 or not isinstance(call.func, cst.Attribute) or call.func.attr.value != "all":
        return None
    if not isinstance(call.func.value, cst.Name):
        return None  # pragma: no cover - qualified names have a Name root
    comparison = call.args[0].value
    if not isinstance(comparison, cst.Comparison) or len(comparison.comparisons) != 1:
        return None  # pragma: no cover - guaranteed by NAL004
    target = comparison.comparisons[0]
    if not isinstance(target.operator, cst.Equal):
        return None  # pragma: no cover - guaranteed by NAL004

    left = comparison.left
    right = target.comparator
    if _is_scalar_literal(left):
        actual, desired = right, left
    elif _is_scalar_literal(right) or allow_unsafe:
        actual, desired = left, right
    else:
        return None

    numpy_alias = call.func.value.value
    testing = cst.Attribute(value=cst.Name(numpy_alias), attr=cst.Name("testing"))
    function = cst.Attribute(value=testing, attr=cst.Name("assert_array_equal"))
    replacement_args = _binary_replacement_args(call, actual, desired)
    if node.msg is not None:
        replacement_args = _append_keyword_args(replacement_args, [("err_msg", node.msg)])
    return cst.Expr(
        value=call.with_changes(func=function, args=replacement_args),
        semicolon=node.semicolon,
    )


def _fix_assertion(
    node: cst.Assert,
    code: str,
    *,
    allow_unsafe: bool,
    numpy_alias: str | None,
) -> cst.Expr | None:
    if code == "NAL001":
        return _fix_allclose(node, allow_unsafe=allow_unsafe)
    if code == "NAL002":
        return _fix_array_equal(node, allow_unsafe=allow_unsafe)
    if code == "NAL003":
        return _fix_all_isclose(node, allow_unsafe=allow_unsafe)
    if code == "NAL004":
        return _fix_numpy_all_equality(node, allow_unsafe=allow_unsafe)
    if code == "NAL005":
        return _fix_equality_all_method(node, allow_unsafe=allow_unsafe, numpy_alias=numpy_alias)
    return None  # pragma: no cover - checker currently emits only NAL001 through NAL005


def _fix_array_equal(node: cst.Assert, *, allow_unsafe: bool) -> cst.Expr | None:
    if not allow_unsafe or not isinstance(node.test, cst.Call):
        return None
    call = node.test
    if not isinstance(call.func, cst.Attribute) or call.func.attr.value != "array_equal":
        return None
    if not isinstance(call.func.value, cst.Name):
        return None  # pragma: no cover - qualified names have a Name root
    arguments = _comparison_arguments(call, parameter_names=("a1", "a2", "equal_nan"))
    if arguments is None:
        return None
    values, preserved_args = arguments
    numpy_alias = call.func.value.value
    testing = cst.Attribute(value=cst.Name(numpy_alias), attr=cst.Name("testing"))
    function = cst.Attribute(value=testing, attr=cst.Name("assert_array_equal"))
    replacement_args = [cst.Arg(values["a1"]), cst.Arg(values["a2"])] if "equal_nan" in values else preserved_args
    if node.msg is not None:
        replacement_args = _append_keyword_args(replacement_args, [("err_msg", node.msg)])
    return cst.Expr(value=call.with_changes(func=function, args=replacement_args), semicolon=node.semicolon)


def _fix_equality_all_method(
    node: cst.Assert,
    *,
    allow_unsafe: bool,
    numpy_alias: str | None,
) -> cst.Expr | None:
    if not allow_unsafe or numpy_alias is None or not isinstance(node.test, cst.Call):
        return None
    call = node.test
    if call.args or not isinstance(call.func, cst.Attribute) or call.func.attr.value != "all":
        return None  # pragma: no cover - guaranteed by NAL005
    comparison = call.func.value
    if not isinstance(comparison, cst.Comparison) or len(comparison.comparisons) != 1:
        return None  # pragma: no cover - guaranteed by NAL005
    target = comparison.comparisons[0]
    if not isinstance(target.operator, cst.Equal):
        return None  # pragma: no cover - guaranteed by NAL005
    testing = cst.Attribute(value=cst.Name(numpy_alias), attr=cst.Name("testing"))
    function = cst.Attribute(value=testing, attr=cst.Name("assert_array_equal"))
    replacement_args = [cst.Arg(comparison.left), cst.Arg(target.comparator)]
    if node.msg is not None:
        replacement_args.append(_keyword_arg("err_msg", node.msg))
    return cst.Expr(value=cst.Call(func=function, args=replacement_args), semicolon=node.semicolon)


def _fix_allclose(node: cst.Assert, *, allow_unsafe: bool) -> cst.Expr | None:
    if not isinstance(node.test, cst.Call):
        return None  # pragma: no cover - guaranteed by NAL001
    if node.msg is not None and not allow_unsafe and not isinstance(node.msg, cst.SimpleString):
        return None
    call = node.test
    if not isinstance(call.func, cst.Attribute) or call.func.attr.value != "allclose":
        return None
    if not isinstance(call.func.value, cst.Name):
        return None  # pragma: no cover - qualified names have a Name root

    return _build_allclose_fix(node, call, numpy_alias=call.func.value.value, allow_unsafe=allow_unsafe)


def _fix_all_isclose(node: cst.Assert, *, allow_unsafe: bool) -> cst.Expr | None:
    if not isinstance(node.test, cst.Call):
        return None  # pragma: no cover - guaranteed by NAL003
    outer_call = node.test
    if len(outer_call.args) != 1 or not isinstance(outer_call.func, cst.Attribute):
        return None
    if outer_call.func.attr.value != "all" or not isinstance(outer_call.func.value, cst.Name):
        return None  # pragma: no cover - qualified names have a Name root
    inner_call = outer_call.args[0].value
    if not isinstance(inner_call, cst.Call) or not isinstance(inner_call.func, cst.Attribute):
        return None
    if inner_call.func.attr.value != "isclose" or not isinstance(inner_call.func.value, cst.Name):
        return None  # pragma: no cover - qualified names have a Name root

    return _build_allclose_fix(
        node,
        inner_call,
        numpy_alias=outer_call.func.value.value,
        allow_unsafe=allow_unsafe,
    )


def _build_allclose_fix(
    node: cst.Assert,
    call: cst.Call,
    *,
    numpy_alias: str,
    allow_unsafe: bool,
) -> cst.Expr | None:
    if node.msg is not None and not allow_unsafe and not isinstance(node.msg, cst.SimpleString):
        return None

    arguments = _comparison_arguments(call, parameter_names=("a", "b", "rtol", "atol", "equal_nan"))
    if arguments is None:
        return None
    values, preserved_args = arguments
    actual = values["a"]
    desired = values["b"]
    if not allow_unsafe and not _is_safe_allclose(actual, desired, values):
        return None

    defaults: tuple[tuple[str, cst.BaseExpression], ...] = (
        ("rtol", cst.Float("1e-5")),
        ("atol", cst.Float("1e-8")),
        ("equal_nan", cst.Name("False")),
    )
    additional_args = [(name, value) for name, value in defaults if name not in values]
    if node.msg is not None:
        additional_args.append(("err_msg", node.msg))
    preserved_args = _append_keyword_args(preserved_args, additional_args)

    testing = cst.Attribute(value=cst.Name(numpy_alias), attr=cst.Name("testing"))
    function = cst.Attribute(value=testing, attr=cst.Name("assert_allclose"))
    return cst.Expr(value=call.with_changes(func=function, args=preserved_args), semicolon=node.semicolon)


def _comparison_arguments(
    call: cst.Call,
    *,
    parameter_names: tuple[str, ...],
) -> tuple[dict[str, cst.BaseExpression], list[cst.Arg]] | None:
    values: dict[str, cst.BaseExpression] = {}
    preserved_args: list[cst.Arg] = []
    positional_index = 0
    for argument in call.args:
        if argument.star:
            return None
        if argument.keyword is None:
            if positional_index >= len(parameter_names):
                return None
            name = parameter_names[positional_index]
            positional_index += 1
            preserved_args.append(argument)
        else:
            name = argument.keyword.value
            if name not in parameter_names:
                return None
            replacement_name = {"a": "actual", "a1": "actual", "a2": "desired", "b": "desired"}.get(name, name)
            preserved_args.append(argument.with_changes(keyword=cst.Name(replacement_name)))
        if name in values:
            return None
        values[name] = argument.value

    if parameter_names[0] not in values or parameter_names[1] not in values:
        return None
    return values, preserved_args


def _is_safe_allclose(
    actual: cst.BaseExpression,
    desired: cst.BaseExpression,
    values: dict[str, cst.BaseExpression],
) -> bool:
    if not (_is_scalar_literal(actual) or _is_scalar_literal(desired) or _same_expression(actual, desired)):
        return False
    for name in ("rtol", "atol"):
        if name in values and not _is_numeric_literal(values[name]):
            return False
    return "equal_nan" not in values or (
        isinstance(values["equal_nan"], cst.Name) and values["equal_nan"].value in {"False", "True"}
    )


def _same_expression(left: cst.BaseExpression, right: cst.BaseExpression) -> bool:
    if not isinstance(left, cst.Name) or not isinstance(right, cst.Name):
        return False
    module = cst.Module(body=[])
    return module.code_for_node(left) == module.code_for_node(right)


def _is_numeric_literal(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Integer | cst.Float | cst.Imaginary) or (
        isinstance(node, cst.UnaryOperation) and isinstance(node.expression, cst.Integer | cst.Float | cst.Imaginary)
    )


def _keyword_arg(
    name: str,
    value: cst.BaseExpression,
    *,
    comma: cst.Comma | cst.MaybeSentinel = cst.MaybeSentinel.DEFAULT,
) -> cst.Arg:
    no_space = cst.SimpleWhitespace("")
    return cst.Arg(
        value=value,
        keyword=cst.Name(name),
        equal=cst.AssignEqual(whitespace_before=no_space, whitespace_after=no_space),
        comma=comma,
    )


def _append_keyword_args(
    arguments: list[cst.Arg],
    additions: list[tuple[str, cst.BaseExpression]],
) -> list[cst.Arg]:
    if not additions:
        return arguments
    if not arguments:
        return [  # pragma: no cover - comparison calls always retain two operands
            _keyword_arg(name, value) for name, value in additions
        ]

    last_argument = arguments[-1]
    if isinstance(last_argument.comma, cst.Comma) and isinstance(
        last_argument.comma.whitespace_after, cst.ParenthesizedWhitespace
    ):
        closing_whitespace = last_argument.comma.whitespace_after
    elif isinstance(last_argument.whitespace_after_arg, cst.ParenthesizedWhitespace):
        closing_whitespace = last_argument.whitespace_after_arg
    else:
        return [*arguments, *(_keyword_arg(name, value) for name, value in additions)]

    final_closing_whitespace = closing_whitespace.with_changes(
        first_line=closing_whitespace.first_line.with_changes(
            whitespace=cst.SimpleWhitespace(""),
            comment=None,
        )
    )
    separator_whitespace = _multiline_argument_whitespace(arguments) or final_closing_whitespace.with_changes(
        last_line=cst.SimpleWhitespace("    "),
    )
    if isinstance(last_argument.whitespace_after_arg, cst.ParenthesizedWhitespace):
        last_argument = last_argument.with_changes(
            comma=cst.Comma(whitespace_after=closing_whitespace.with_changes(last_line=separator_whitespace.last_line)),
            whitespace_after_arg=cst.SimpleWhitespace(""),
        )
    elif isinstance(last_argument.comma, cst.Comma):
        last_argument = last_argument.with_changes(
            comma=last_argument.comma.with_changes(
                whitespace_after=closing_whitespace.with_changes(last_line=separator_whitespace.last_line)
            )
        )
    updated_arguments = [*arguments[:-1], last_argument]
    for index, (name, value) in enumerate(additions):
        whitespace_after = final_closing_whitespace if index == len(additions) - 1 else separator_whitespace
        updated_arguments.append(_keyword_arg(name, value, comma=cst.Comma(whitespace_after=whitespace_after)))
    return updated_arguments


def _multiline_argument_whitespace(arguments: list[cst.Arg]) -> cst.ParenthesizedWhitespace | None:
    for argument in arguments[:-1]:
        if not isinstance(argument.comma, cst.Comma):
            continue  # pragma: no cover - all non-final arguments require a comma
        whitespace = argument.comma.whitespace_after
        if not isinstance(whitespace, cst.ParenthesizedWhitespace):
            continue
        return whitespace.with_changes(
            first_line=whitespace.first_line.with_changes(
                whitespace=cst.SimpleWhitespace(""),
                comment=None,
            )
        )
    return None


def _binary_replacement_args(
    call: cst.Call,
    actual: cst.BaseExpression,
    desired: cst.BaseExpression,
) -> list[cst.Arg]:
    original_argument = call.args[0]
    if not isinstance(call.whitespace_before_args, cst.ParenthesizedWhitespace):
        return [cst.Arg(actual), cst.Arg(desired)]

    opening_whitespace = call.whitespace_before_args
    separator_whitespace = opening_whitespace.with_changes(
        first_line=opening_whitespace.first_line.with_changes(
            whitespace=cst.SimpleWhitespace(""),
            comment=None,
        )
    )
    return [
        cst.Arg(actual, comma=cst.Comma(whitespace_after=separator_whitespace)),
        cst.Arg(
            desired,
            comma=original_argument.comma,
            whitespace_after_arg=original_argument.whitespace_after_arg,
        ),
    ]


def _wrap_long_call(node: cst.Expr, *, starting_column: int) -> cst.Expr:
    rendered = cst.Module(body=[]).code_for_node(node)
    if "\n" in rendered or "\r" in rendered or starting_column + len(rendered) <= _MAX_LINE_LENGTH:
        return node
    if not isinstance(node.value, cst.Call):  # pragma: no cover - every fixer produces a call expression
        return node

    arguments: list[cst.Arg] = []
    for index, argument in enumerate(node.value.args):
        arguments.append(
            argument.with_changes(
                comma=cst.Comma(whitespace_after=_call_line_whitespace(closing=index == len(node.value.args) - 1)),
                whitespace_after_arg=cst.SimpleWhitespace(""),
            )
        )
    return node.with_changes(
        value=node.value.with_changes(
            args=arguments,
            whitespace_before_args=_call_line_whitespace(closing=False),
        )
    )


def _call_line_whitespace(*, closing: bool) -> cst.ParenthesizedWhitespace:
    return cst.ParenthesizedWhitespace(
        first_line=cst.TrailingWhitespace(newline=cst.Newline()),
        indent=True,
        last_line=cst.SimpleWhitespace("" if closing else "    "),
    )


def _is_scalar_literal(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.Integer | cst.Float | cst.Imaginary | cst.SimpleString):
        return True
    if isinstance(node, cst.Name):
        return node.value in {"False", "None", "True"}
    return isinstance(node, cst.UnaryOperation) and isinstance(node.expression, cst.Integer | cst.Float | cst.Imaginary)
