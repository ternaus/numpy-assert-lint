# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## 0.2.0 — 2026-07-31

- Add concrete-syntax-tree fixes that preserve source formatting, comments, encoding, and line endings.
- Add `--diff`, `--fix`, and the explicit `--unsafe-fixes` opt-in.
- Add the `numpy-assert-lint-fix` pre-commit hook for conservative automatic fixes.
- Preserve `np.allclose` tolerances and NaN handling when converting to `np.testing.assert_allclose`.

## 0.1.0 — 2026-07-31

- Detect four explicit NumPy assertion patterns.
- Provide an opt-in rule for ambiguous `.all()` equality checks.
- Support import aliases, `# noqa`, CLI rule selection, and `pyproject.toml` configuration.
- Publish a reusable pre-commit hook.
