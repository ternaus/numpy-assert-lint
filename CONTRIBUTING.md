# Contributing

## Set up the development environment

The project uses Python 3.10 or newer. Install the development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Run the checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov=numpy_assert_lint --cov-report=term-missing
uv run pre-commit validate-manifest
uv run python -m build
```

Add a focused test for each new behavior. The test should exercise `check_source()` or the CLI, fail before the implementation change, and pass after the smallest corresponding code change.

## Add or change a rule

Keep default rules limited to patterns that identify NumPy calls directly. A rule that relies on an inferred array type should be opt-in unless the checker can avoid false positives without importing or executing user code.

Do not add an automatic fix when the replacement changes tolerances, NaN handling, broadcasting, shape checks, or dtype checks.
