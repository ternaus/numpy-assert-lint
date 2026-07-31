"""Tests for NumPy assertion fixes."""

import pytest

from numpy_assert_lint.fixer import fix_source


def test_fixes_scalar_numpy_all_equality() -> None:
    result = fix_source("import numpy as np\nassert np.all(actual == 0)\n")

    assert result.source == "import numpy as np\nnp.testing.assert_array_equal(actual, 0)\n"
    assert result.fixed_codes == ("NAL004",)
    assert result.skipped_unsafe_codes == ()


def test_fixes_array_equality_when_unsafe_fixes_are_enabled() -> None:
    result = fix_source(
        "import numpy as np\nassert np.all(actual == expected)\n",
        allow_unsafe=True,
    )

    assert result.source == "import numpy as np\nnp.testing.assert_array_equal(actual, expected)\n"
    assert result.fixed_codes == ("NAL004",)
    assert result.skipped_unsafe_codes == ()


def test_fixes_scalar_allclose_and_preserves_numpy_defaults() -> None:
    result = fix_source("import numpy as np\nassert np.allclose(actual, 0)\n")

    assert result.source == (
        "import numpy as np\nnp.testing.assert_allclose(actual, 0, rtol=1e-5, atol=1e-8, equal_nan=False)\n"
    )
    assert result.fixed_codes == ("NAL001",)


def test_preserves_literal_assert_message_as_numpy_error_message() -> None:
    result = fix_source('import numpy as np\nassert np.allclose(actual, 0), "pixels differ"\n')

    assert result.source == (
        "import numpy as np\n"
        'np.testing.assert_allclose(actual, 0, rtol=1e-5, atol=1e-8, equal_nan=False, err_msg="pixels differ")\n'
    )


def test_wraps_fix_that_would_exceed_120_characters() -> None:
    result = fix_source(
        "import numpy as np\n"
        "assert np.allclose(transformed[transformed > 0.9], 1.0), "
        '"Salt pixels should be exactly 1.0 for float images"\n'
    )

    assert result.source == (
        "import numpy as np\n"
        "np.testing.assert_allclose(\n"
        "    transformed[transformed > 0.9],\n"
        "    1.0,\n"
        "    rtol=1e-5,\n"
        "    atol=1e-8,\n"
        "    equal_nan=False,\n"
        '    err_msg="Salt pixels should be exactly 1.0 for float images",\n'
        ")\n"
    )


def test_fixes_scalar_numpy_all_isclose() -> None:
    result = fix_source("import numpy as np\nassert np.all(np.isclose(actual, 0, atol=1e-4))\n")

    assert result.source == (
        "import numpy as np\nnp.testing.assert_allclose(actual, 0, atol=1e-4, rtol=1e-5, equal_nan=False)\n"
    )
    assert result.fixed_codes == ("NAL003",)


def test_fixes_array_equal_when_unsafe_fixes_are_enabled() -> None:
    result = fix_source(
        "import numpy as np\nassert np.array_equal(actual, expected)\n",
        allow_unsafe=True,
    )

    assert result.source == "import numpy as np\nnp.testing.assert_array_equal(actual, expected)\n"
    assert result.fixed_codes == ("NAL002",)


def test_fixes_equality_all_method_with_visible_numpy_alias() -> None:
    result = fix_source(
        "import numpy as np\nassert (actual == expected).all()\n",
        allow_unsafe=True,
    )

    assert result.source == "import numpy as np\nnp.testing.assert_array_equal(actual, expected)\n"
    assert result.fixed_codes == ("NAL005",)


def test_method_fix_uses_numpy_alias_visible_inside_function() -> None:
    result = fix_source(
        "def test_arrays():\n    import numpy as xp\n    assert (actual == expected).all()\n",
        allow_unsafe=True,
    )

    assert result.source == (
        "def test_arrays():\n    import numpy as xp\n    xp.testing.assert_array_equal(actual, expected)\n"
    )


def test_method_fix_does_not_use_a_shadowed_numpy_alias() -> None:
    source = "import numpy as np\ndef test_arrays(np):\n    assert (actual == expected).all()\n"

    result = fix_source(source, allow_unsafe=True)

    assert result.source == source
    assert result.fixed_codes == ()


def test_method_fix_does_not_use_a_class_local_numpy_alias() -> None:
    source = (
        "class TestArrays:\n"
        "    import numpy as xp\n"
        "    def test_arrays(self):\n"
        "        assert (actual == expected).all()\n"
    )

    result = fix_source(source, allow_unsafe=True)

    assert result.source == source


def test_method_fix_uses_numpy_alias_in_class_body() -> None:
    result = fix_source(
        "class TestArrays:\n    import numpy as xp\n    assert (actual == expected).all()\n",
        allow_unsafe=True,
    )

    assert result.source == (
        "class TestArrays:\n    import numpy as xp\n    xp.testing.assert_array_equal(actual, expected)\n"
    )


@pytest.mark.parametrize(
    "shadowing_statement",
    [
        "import pandas as np",
        "from pandas import *",
        "np = object()",
    ],
)
def test_method_fix_respects_alias_invalidation(shadowing_statement: str) -> None:
    source = f"import numpy as np\n{shadowing_statement}\nassert (actual == expected).all()\n"

    result = fix_source(source, allow_unsafe=True)

    assert result.source == source


@pytest.mark.parametrize("parameter", ["*np", "**np"])
def test_method_fix_respects_variadic_parameter_shadowing(parameter: str) -> None:
    source = f"import numpy as np\nasync def test_arrays({parameter}):\n    assert (actual == expected).all()\n"

    result = fix_source(source, allow_unsafe=True)

    assert result.source == source


def test_only_fixes_enabled_rule_codes() -> None:
    result = fix_source(
        "import numpy as np\nassert np.allclose(actual, 0)\nassert np.all(actual == 0)\n",
        enabled_codes={"NAL004"},
    )

    assert result.source == (
        "import numpy as np\nassert np.allclose(actual, 0)\nnp.testing.assert_array_equal(actual, 0)\n"
    )
    assert result.fixed_codes == ("NAL004",)


def test_preserves_multiline_layout_and_argument_comments() -> None:
    result = fix_source("import numpy as np\nassert np.allclose(\n    actual,  # computed value\n    0,\n)\n")

    assert result.source == (
        "import numpy as np\n"
        "np.testing.assert_allclose(\n"
        "    actual,  # computed value\n"
        "    0,\n"
        "    rtol=1e-5,\n"
        "    atol=1e-8,\n"
        "    equal_nan=False,\n"
        ")\n"
    )


def test_array_equal_fix_preserves_multiline_layout_and_comments() -> None:
    result = fix_source(
        "import numpy as np\nassert np.array_equal(\n    actual,  # computed value\n    expected,\n)\n",
        allow_unsafe=True,
    )

    assert result.source == (
        "import numpy as np\nnp.testing.assert_array_equal(\n    actual,  # computed value\n    expected,\n)\n"
    )


def test_numpy_all_fix_preserves_multiline_layout_and_comment() -> None:
    result = fix_source("import numpy as np\nassert np.all(\n    actual == 0  # pixels\n)\n")

    assert result.source == ("import numpy as np\nnp.testing.assert_array_equal(\n    actual,\n    0  # pixels\n)\n")


def test_preserves_literal_message_for_scalar_array_equality() -> None:
    result = fix_source('import numpy as np\nassert np.all(actual == 0), "nonzero pixels"\n')

    assert result.source == ('import numpy as np\nnp.testing.assert_array_equal(actual, 0, err_msg="nonzero pixels")\n')


def test_reports_fixable_but_unsafe_rule() -> None:
    source = "import numpy as np\nassert np.all(actual == expected)\n"

    result = fix_source(source)

    assert result.source == source
    assert result.fixed_codes == ()
    assert result.skipped_unsafe_codes == ("NAL004",)


def test_does_not_rewrite_direct_function_imports() -> None:
    source = "from numpy import allclose\nassert allclose(actual, 0)\n"

    result = fix_source(source, allow_unsafe=True)

    assert result.source == source
    assert result.fixed_codes == ()
    assert result.skipped_unsafe_codes == ()


def test_fix_is_idempotent() -> None:
    first = fix_source("import numpy as np\nassert np.all(actual == 0)\n")

    second = fix_source(first.source)

    assert second.source == first.source
    assert second.fixed_codes == ()


def test_preserves_explicit_allclose_parameters_and_keyword_operands() -> None:
    result = fix_source(
        "import numpy as xp\nassert xp.allclose(a=actual, b=0, rtol=-1e-4, atol=2e-5, equal_nan=True)\n"
    )

    assert result.source == (
        "import numpy as xp\n"
        "xp.testing.assert_allclose(actual=actual, desired=0, rtol=-1e-4, atol=2e-5, equal_nan=True)\n"
    )


def test_requires_unsafe_mode_for_array_allclose() -> None:
    source = "import numpy as np\nassert np.allclose(actual, expected)\n"

    safe_result = fix_source(source)
    unsafe_result = fix_source(source, allow_unsafe=True)

    assert safe_result.source == source
    assert safe_result.skipped_unsafe_codes == ("NAL001",)
    assert unsafe_result.source == (
        "import numpy as np\nnp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)\n"
    )


def test_unsafe_fix_can_transfer_dynamic_assert_message() -> None:
    source = "import numpy as np\nassert np.allclose(actual, expected), failure_message()\n"

    safe_result = fix_source(source)
    result = fix_source(source, allow_unsafe=True)

    assert safe_result.skipped_unsafe_codes == ("NAL001",)
    assert result.source == (
        "import numpy as np\n"
        "np.testing.assert_allclose("
        "actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False, err_msg=failure_message()"
        ")\n"
    )


def test_does_not_apply_safe_fix_with_dynamic_tolerances() -> None:
    source = "import numpy as np\nassert np.allclose(actual, 0, rtol=tolerance)\n"

    result = fix_source(source)

    assert result.source == source
    assert result.skipped_unsafe_codes == ("NAL001",)


def test_unsafe_array_equal_drops_equal_nan_and_preserves_message() -> None:
    result = fix_source(
        'import numpy as np\nassert np.array_equal(actual, expected, equal_nan=True), "arrays differ"\n',
        allow_unsafe=True,
    )

    assert result.source == (
        'import numpy as np\nnp.testing.assert_array_equal(actual, expected, err_msg="arrays differ")\n'
    )


def test_unsafe_method_fix_uses_original_numpy_alias_and_message() -> None:
    result = fix_source(
        'import numpy as xp\nassert (actual == expected).all(), "arrays differ"\n',
        allow_unsafe=True,
    )

    assert result.source == (
        'import numpy as xp\nxp.testing.assert_array_equal(actual, expected, err_msg="arrays differ")\n'
    )


def test_method_fix_requires_a_module_numpy_import() -> None:
    source = "assert (actual == expected).all()\n"

    result = fix_source(source, allow_unsafe=True)

    assert result.source == source
    assert result.fixed_codes == ()


@pytest.mark.parametrize("scalar", ["True", "-1"])
def test_safe_equality_supports_literal_scalar_forms(scalar: str) -> None:
    result = fix_source(f"import numpy as np\nassert np.all(actual == {scalar})\n")

    assert result.source == f"import numpy as np\nnp.testing.assert_array_equal(actual, {scalar})\n"


def test_safe_equality_normalizes_scalar_on_left() -> None:
    result = fix_source("import numpy as np\nassert np.all(0 == actual)\n")

    assert result.source == "import numpy as np\nnp.testing.assert_array_equal(actual, 0)\n"


def test_identical_allclose_operands_are_safe_to_fix() -> None:
    result = fix_source("import numpy as np\nassert np.allclose(actual, actual)\n")

    assert result.fixed_codes == ("NAL001",)


def test_repeated_calls_are_not_treated_as_stable_identical_operands() -> None:
    source = "import numpy as np\nassert np.allclose(make_array(), make_array())\n"

    result = fix_source(source)

    assert result.source == source
    assert result.skipped_unsafe_codes == ("NAL001",)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nassert np.allclose(*values)\n",
        "import numpy as np\nassert np.allclose(actual, 0, where=mask)\n",
        "import numpy as np\nassert np.allclose(actual)\n",
        "import numpy as np\nassert np.allclose(actual, 0, 1e-5, 1e-8, False, extra)\n",
        "import numpy as np\nassert np.allclose(actual, 0, a=other)\n",
        "import numpy as np\nassert np.array_equal(*values)\n",
        "from numpy import array_equal\nassert array_equal(actual, expected)\n",
        "import numpy as np\nfrom numpy import isclose\nassert np.all(isclose(actual, 0))\n",
        "import numpy as np\nfrom numpy import all as arrays_all\nassert arrays_all(np.isclose(actual, 0))\n",
        "from numpy import all as arrays_all\nassert arrays_all(actual == 0)\n",
    ],
)
def test_leaves_unsupported_call_forms_unchanged(source: str) -> None:
    result = fix_source(source, allow_unsafe=True)

    assert result.source == source
    assert result.fixed_codes == ()


def test_keeps_complete_allclose_argument_list_without_adding_defaults() -> None:
    result = fix_source("import numpy as np\nassert np.allclose(actual, 0, 1e-5, 1e-8, False)\n")

    assert result.source == "import numpy as np\nnp.testing.assert_allclose(actual, 0, 1e-5, 1e-8, False)\n"


def test_default_mode_reports_array_equal_as_unsafe() -> None:
    source = "import numpy as np\nassert np.array_equal(actual, expected)\n"

    result = fix_source(source)

    assert result.source == source
    assert result.skipped_unsafe_codes == ("NAL002",)


def test_dynamic_messages_require_unsafe_mode() -> None:
    source = "import numpy as np\nassert np.all(actual == 0), failure_message()\n"

    safe_result = fix_source(source)
    unsafe_result = fix_source(source, allow_unsafe=True)

    assert safe_result.source == source
    assert safe_result.skipped_unsafe_codes == ("NAL004",)
    assert unsafe_result.fixed_codes == ("NAL004",)


def test_dynamic_message_on_all_isclose_requires_unsafe_mode() -> None:
    source = "import numpy as np\nassert np.all(np.isclose(actual, 0)), failure_message()\n"

    result = fix_source(source)

    assert result.source == source
    assert result.skipped_unsafe_codes == ("NAL003",)


def test_indents_added_arguments_for_compact_multiline_call() -> None:
    result = fix_source("import numpy as np\nassert np.allclose(actual, 0,\n)\n")

    assert result.source == (
        "import numpy as np\n"
        "np.testing.assert_allclose(actual, 0,\n"
        "    rtol=1e-5,\n"
        "    atol=1e-8,\n"
        "    equal_nan=False,\n"
        ")\n"
    )


def test_adds_arguments_to_multiline_call_without_trailing_comma() -> None:
    result = fix_source(
        'import numpy as np\nassert np.allclose(\n    actual,\n    0  # scalar comparison\n), "arrays differ"\n'
    )

    assert result.source == (
        "import numpy as np\n"
        "np.testing.assert_allclose(\n"
        "    actual,\n"
        "    0,  # scalar comparison\n"
        "    rtol=1e-5,\n"
        "    atol=1e-8,\n"
        "    equal_nan=False,\n"
        '    err_msg="arrays differ",\n'
        ")\n"
    )
