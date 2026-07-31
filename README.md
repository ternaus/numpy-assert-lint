# numpy-assert-lint

[![CI](https://github.com/Ternaus/numpy-assert-lint/actions/workflows/ci.yml/badge.svg)](https://github.com/Ternaus/numpy-assert-lint/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`numpy-assert-lint` finds NumPy comparisons that lose useful failure diagnostics inside plain `assert` statements. It runs as a pre-commit hook or standalone CLI and does not import NumPy or execute checked code.

```python
# Reported as NAL001
assert np.allclose(actual, expected)

# Reports mismatched values and their locations
np.testing.assert_allclose(
    actual,
    expected,
    rtol=1e-5,
    atol=1e-8,
    equal_nan=False,
)
```

## Add the pre-commit hook

Add the repository to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/Ternaus/numpy-assert-lint
    rev: v0.1.0
    hooks:
      - id: numpy-assert-lint
        files: ^tests/
```

Run it against the existing test suite:

```bash
pre-commit run numpy-assert-lint --all-files
```

Pre-commit installs the checker in an isolated environment. The checked repository does not need to list `numpy-assert-lint` as a dependency.

## Rules

| Code | Reported pattern | Suggested assertion | Default |
| --- | --- | --- | --- |
| `NAL001` | `assert np.allclose(actual, expected)` | `np.testing.assert_allclose(actual, expected, ...)` | Yes |
| `NAL002` | `assert np.array_equal(actual, expected)` | `np.testing.assert_array_equal(actual, expected)` | Yes |
| `NAL003` | `assert np.all(np.isclose(actual, expected))` | `np.testing.assert_allclose(actual, expected, ...)` | Yes |
| `NAL004` | `assert np.all(actual == expected)` | `np.testing.assert_array_equal(actual, expected)` | Yes |
| `NAL005` | `assert (actual == expected).all()` | Verify the array type, then use its diagnostic assertion | No |

`NAL005` is opt-in because the method form does not identify the array library. NumPy, pandas, PyTorch, JAX, and other objects can expose `.all()` with different comparison APIs.

The checker resolves common import forms:

```python
import numpy
import numpy as np
import numpy as xp
from numpy import allclose
from numpy import allclose as arrays_close
```

It also tracks basic shadowing by assignments, function parameters, functions, and classes.

## Review every replacement

The checker reports diagnostics without rewriting code. NumPy's comparison functions and testing assertions have different defaults:

- [`np.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) defaults to `rtol=1e-5`, `atol=1e-8`, and `equal_nan=False`. It uses broadcasting.
- [`np.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html) defaults to `rtol=1e-7`, `atol=0`, and `equal_nan=True`. It rejects broadcasting between non-scalar operands.
- [`np.array_equal`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html) defaults to `equal_nan=False`, while [`np.testing.assert_array_equal`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_array_equal.html) treats NaNs at the same positions as equal.

A mechanical replacement could change whether a test passes. Choose tolerances, NaN handling, shape checks, and dtype checks from the contract under test.

## Configure enabled rules

The four rules with explicit NumPy calls are enabled by default. Configure rule codes or prefixes in `pyproject.toml`:

```toml
[tool.numpy-assert-lint]
select = ["NAL001", "NAL002", "NAL003", "NAL004", "NAL005"]
ignore = []
```

Command-line options override `pyproject.toml`:

```bash
numpy-assert-lint --select NAL001,NAL002 tests/
numpy-assert-lint --ignore NAL003 tests/test_metrics.py
numpy-assert-lint --config config/pyproject.toml tests/
```

The CLI scans Python files recursively when given a directory. With no paths, it scans the current directory and skips hidden directories, `build`, `dist`, `node_modules`, and `site-packages`.

Exit codes are stable:

- `0`: no enabled violations;
- `1`: one or more enabled violations;
- `2`: invalid configuration, missing input, invalid Python source or encoding, or CLI usage error.

## Suppress an intentional assertion

Use a rule-specific `# noqa` comment:

```python
assert np.allclose(actual, expected)  # noqa: NAL001
```

Bare `# noqa` suppresses every `numpy-assert-lint` diagnostic on the assertion. For multiline assertions, place the comment on any line in the assertion.

If Ruff validates or removes `# noqa` comments, register the external prefix:

```toml
[tool.ruff.lint]
external = ["NAL"]
```

## Run as a standalone tool

Install the tagged GitHub release with `uv`:

```bash
uv tool install "git+https://github.com/Ternaus/numpy-assert-lint@v0.1.0"
numpy-assert-lint tests/
```

The package supports Python 3.10 through 3.14. NumPy is not a runtime dependency.

## Develop the checker

```bash
git clone https://github.com/Ternaus/numpy-assert-lint.git
cd numpy-assert-lint
uv sync --extra dev
uv run pytest --cov=numpy_assert_lint --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pre-commit validate-manifest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for rule design constraints and the complete validation commands.

## License

`numpy-assert-lint` is available under the [MIT License](LICENSE).
