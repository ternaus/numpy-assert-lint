import pytest

from numpy_assert_lint.checker import check_source


def test_flags_numpy_allclose_assert() -> None:
    diagnostics = check_source(
        "import numpy as np\nassert np.allclose(actual, expected)\n",
        filename="tests/test_example.py",
    )

    assert [(item.line, item.column, item.code) for item in diagnostics] == [(2, 1, "NAL001")]


def test_flags_numpy_array_equal_assert() -> None:
    diagnostics = check_source(
        "import numpy as np\nassert np.array_equal(actual, expected)\n",
        filename="tests/test_example.py",
    )

    assert [(item.line, item.column, item.code) for item in diagnostics] == [(2, 1, "NAL002")]


def test_flags_numpy_all_isclose_assert() -> None:
    diagnostics = check_source(
        "import numpy as np\nassert np.all(np.isclose(actual, expected))\n",
        filename="tests/test_example.py",
    )

    assert [(item.line, item.column, item.code) for item in diagnostics] == [(2, 1, "NAL003")]


def test_flags_numpy_all_equality_assert() -> None:
    diagnostics = check_source(
        "import numpy as np\nassert np.all(actual == expected)\n",
        filename="tests/test_example.py",
    )

    assert [(item.line, item.column, item.code) for item in diagnostics] == [(2, 1, "NAL004")]


def test_flags_equality_all_method_assert() -> None:
    diagnostics = check_source(
        "assert (actual == expected).all()\n",
        filename="tests/test_example.py",
    )

    assert [(item.line, item.column, item.code) for item in diagnostics] == [(1, 1, "NAL005")]


def test_resolves_functions_imported_from_numpy() -> None:
    diagnostics = check_source(
        "from numpy import allclose as arrays_close\nassert arrays_close(actual, expected)\n",
        filename="tests/test_example.py",
    )

    assert [(item.line, item.column, item.code) for item in diagnostics] == [(2, 1, "NAL001")]


def test_ignores_numpy_alias_shadowed_by_function_parameter() -> None:
    diagnostics = check_source(
        "import numpy as np\ndef compare(np):\n    assert np.allclose(actual, expected)\n",
        filename="tests/test_example.py",
    )

    assert diagnostics == []


def test_ignores_numpy_alias_shadowed_by_assignment() -> None:
    diagnostics = check_source(
        "import numpy as np\nnp = custom_backend\nassert np.allclose(actual, expected)\n",
        filename="tests/test_example.py",
    )

    assert diagnostics == []


def test_honors_rule_specific_noqa() -> None:
    diagnostics = check_source(
        "import numpy as np\nassert np.allclose(actual, expected)  # noqa: NAL001\n",
        filename="tests/test_example.py",
    )

    assert diagnostics == []


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        ("import numpy as xp\nassert xp.allclose(a, b)\n", "NAL001"),
        ("import numpy\nassert numpy.array_equal(a, b)\n", "NAL002"),
        ("from numpy import all, isclose\nassert all(isclose(a, b))\n", "NAL003"),
    ],
)
def test_resolves_supported_numpy_import_styles(source: str, expected_code: str) -> None:
    diagnostics = check_source(source)

    assert [item.code for item in diagnostics] == [expected_code]


@pytest.mark.parametrize(
    "source",
    [
        "assert scalar == expected\n",
        "from custom import allclose\nassert allclose(a, b)\n",
        "import numpy as np\nassert not np.allclose(a, b)\n",
        "import numpy as np\nassert np.all(a == b, axis=0)\n",
        "import numpy as np\nassert np.all(a != b)\n",
        "import numpy as np\nnp.testing.assert_allclose(a, b)\n",
        "import numpy as np\nresult = np.allclose(a, b)\n",
        "assert factory().allclose(a, b)\n",
        "# explain why this assertion matters\nassert scalar\n",
    ],
)
def test_ignores_assertions_without_a_supported_replacement(source: str) -> None:
    assert check_source(source) == []


@pytest.mark.parametrize(
    "comment",
    ["# noqa", "# noqa: NAL001, F401"],
)
def test_honors_noqa_variants(comment: str) -> None:
    source = f"import numpy as np\nassert np.allclose(\n    actual,\n    expected,\n)  {comment}\n"

    assert check_source(source) == []


def test_does_not_treat_noqa_inside_a_string_as_suppression() -> None:
    diagnostics = check_source('import numpy as np\nassert np.allclose(a, b), "# noqa: NAL001"\n')

    assert [item.code for item in diagnostics] == ["NAL001"]


def test_ignores_numpy_alias_rebound_by_function_definition() -> None:
    diagnostics = check_source(
        "import numpy as np\ndef np():\n    pass\nassert np.allclose(a, b)\n",
    )

    assert diagnostics == []


def test_ignores_numpy_alias_rebound_by_class_definition() -> None:
    diagnostics = check_source(
        "import numpy as np\nclass np:\n    pass\nassert np.allclose(a, b)\n",
    )

    assert diagnostics == []


def test_ignores_aliases_shadowed_by_async_variadic_parameters() -> None:
    diagnostics = check_source(
        "import numpy\nimport numpy as np\n"
        "async def compare(*np, **numpy):\n"
        "    assert np.allclose(a, b)\n"
        "    assert numpy.array_equal(a, b)\n",
    )

    assert diagnostics == []


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nimport custom as np\nassert np.allclose(a, b)\n",
        "from numpy import allclose\nfrom custom import allclose\nassert allclose(a, b)\n",
        "import numpy.linalg as np\nassert np.allclose(a, b)\n",
        "import numpy as np\nfrom custom import *\nassert np.allclose(a, b)\n",
        "from .numpy import allclose\nassert allclose(a, b)\n",
    ],
)
def test_ignores_numpy_aliases_shadowed_by_other_imports(source: str) -> None:
    assert check_source(source) == []
