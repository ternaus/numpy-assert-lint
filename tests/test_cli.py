"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from numpy_assert_lint.cli import main

_ERROR_EXIT_CODE = 2


def test_cli_prints_diagnostics_and_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.allclose(actual, expected)\n", encoding="utf-8")

    exit_code = main([str(test_file)])

    assert exit_code == 1
    assert capsys.readouterr().out == (
        f"{test_file}:2:1: NAL001 Prefer np.testing.assert_allclose() for diagnostic output.\n"
    )


def test_cli_lints_directories_recursively(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.array_equal(actual, expected)\n", encoding="utf-8")
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    hidden_file = hidden_dir / "test_hidden.py"
    hidden_file.write_text("import numpy as np\nassert np.allclose(actual, expected)\n", encoding="utf-8")

    exit_code = main([str(tmp_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"{test_file}:2:1: NAL002" in output
    assert str(hidden_file) not in output


def test_cli_reports_syntax_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_broken.py"
    test_file.write_text("assert (\n", encoding="utf-8")

    exit_code = main([str(test_file)])

    assert exit_code == _ERROR_EXIT_CODE
    assert f"{test_file}:1:8: NAL900 SyntaxError:" in capsys.readouterr().err


def test_cli_can_ignore_a_rule(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.allclose(actual, expected)\n", encoding="utf-8")

    exit_code = main(["--ignore", "NAL001", str(test_file)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_cli_can_select_rules(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text(
        "import numpy as np\nassert np.allclose(a, b)\nassert np.array_equal(a, b)\n",
        encoding="utf-8",
    )

    exit_code = main(["--select", "NAL002", str(test_file)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "NAL001" not in output
    assert "NAL002" in output


def test_cli_reads_rule_configuration_from_pyproject(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.allclose(a, b)\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.numpy-assert-lint]\nignore = ["NAL001"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main([str(test_file)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_cli_reports_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])

    assert raised.value.code == 0
    assert capsys.readouterr().out == "numpy-assert-lint 0.2.0\n"


def test_cli_does_not_enable_ambiguous_method_rule_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("assert (actual == expected).all()\n", encoding="utf-8")

    exit_code = main([str(test_file)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_cli_reports_invalid_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.numpy-assert-lint]\nselect = "NAL001"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == (
        "pyproject.toml: NAL901 Invalid configuration: select must be an array of strings\n"
    )


def test_cli_reports_non_table_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool]\nnumpy-assert-lint = "invalid"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == (
        "pyproject.toml: NAL901 Invalid configuration: tool.numpy-assert-lint must be a table\n"
    )


def test_cli_reports_non_table_tool_section(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text('tool = "invalid"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == ("pyproject.toml: NAL901 Invalid configuration: tool must be a table\n")


def test_cli_reports_invalid_toml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("invalid = [\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == _ERROR_EXIT_CODE
    assert "pyproject.toml: NAL901 Invalid configuration:" in capsys.readouterr().err


def test_cli_reports_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_file = tmp_path / "missing.py"

    exit_code = main([str(missing_file)])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == f"{missing_file}: NAL902 File not found\n"


def test_cli_works_without_a_pyproject(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_cli_reports_missing_explicit_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_config = tmp_path / "missing.toml"
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("assert True\n", encoding="utf-8")

    exit_code = main(["--config", str(missing_config), str(test_file)])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == f"{missing_config}: NAL901 Configuration file not found\n"


def test_cli_reports_source_encoding_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_invalid_encoding.py"
    test_file.write_bytes(b"\xff")

    exit_code = main([str(test_file)])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == f"{test_file}: NAL903 Could not read file: invalid or missing encoding\n"


def test_cli_reports_file_read_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_file = tmp_path / "test_unreadable.py"
    test_file.write_text("assert True\n", encoding="utf-8")

    def raise_read_error(_path: Path) -> None:
        message = "permission denied"
        raise OSError(message)

    monkeypatch.setattr("numpy_assert_lint.cli._read_source", raise_read_error)

    exit_code = main([str(test_file)])

    assert exit_code == _ERROR_EXIT_CODE
    assert capsys.readouterr().err == f"{test_file}: NAL903 Could not read file: permission denied\n"


def test_cli_fix_rewrites_safe_violations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.all(actual == 0)\n", encoding="utf-8")

    exit_code = main(["--fix", str(test_file)])

    assert exit_code == 1
    assert test_file.read_text(encoding="utf-8") == ("import numpy as np\nnp.testing.assert_array_equal(actual, 0)\n")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{test_file}: Fixed 1 violation.\n"


def test_cli_reuses_checker_diagnostics_for_fixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.all(actual == 0)\n", encoding="utf-8")

    def fail_if_rechecked(_source: str, *, filename: str) -> None:
        pytest.fail(f"fixer rechecked diagnostics for {filename}")

    monkeypatch.setattr("numpy_assert_lint.fixer.check_source", fail_if_rechecked)

    exit_code = main(["--fix", str(test_file)])

    assert exit_code == 1


def test_cli_fix_reports_violations_without_safe_fixes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_file = tmp_path / "test_arrays.py"
    source = "import numpy as np\nassert np.array_equal(actual, expected)\n"
    test_file.write_text(source, encoding="utf-8")

    exit_code = main(["--fix", str(test_file)])

    assert exit_code == 1
    assert test_file.read_text(encoding="utf-8") == source
    captured = capsys.readouterr()
    assert captured.out == (f"{test_file}:2:1: NAL002 Prefer np.testing.assert_array_equal() for diagnostic output.\n")
    assert captured.err == ""


def test_cli_can_apply_unsafe_fixes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_arrays.py"
    test_file.write_text("import numpy as np\nassert np.array_equal(actual, expected)\n", encoding="utf-8")

    exit_code = main(["--fix", "--unsafe-fixes", str(test_file)])

    assert exit_code == 1
    assert test_file.read_text(encoding="utf-8") == (
        "import numpy as np\nnp.testing.assert_array_equal(actual, expected)\n"
    )
    assert capsys.readouterr().err == f"{test_file}: Fixed 1 violation.\n"


def test_cli_diff_prints_patch_without_changing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_arrays.py"
    source = "import numpy as np\nassert np.all(actual == 0)\n"
    test_file.write_text(source, encoding="utf-8")

    exit_code = main(["--diff", str(test_file)])

    assert exit_code == 1
    assert test_file.read_text(encoding="utf-8") == source
    assert capsys.readouterr().out == (
        f"--- {test_file}:before\n"
        f"+++ {test_file}:after\n"
        "@@ -1,2 +1,2 @@\n"
        " import numpy as np\n"
        "-assert np.all(actual == 0)\n"
        "+np.testing.assert_array_equal(actual, 0)\n"
    )


def test_cli_fix_preserves_source_encoding_and_crlf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "test_latin1.py"
    test_file.write_bytes(b"# coding: latin-1\r\nimport numpy as np\r\nassert np.all(actual == 0)\r\n# caf\xe9\r\n")

    exit_code = main(["--fix", str(test_file)])

    assert exit_code == 1
    assert test_file.read_bytes() == (
        b"# coding: latin-1\r\nimport numpy as np\r\nnp.testing.assert_array_equal(actual, 0)\r\n# caf\xe9\r\n"
    )
    assert capsys.readouterr().err == f"{test_file}: Fixed 1 violation.\n"


def test_cli_rejects_unsafe_fixes_without_an_output_mode(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--unsafe-fixes"])

    assert raised.value.code == _ERROR_EXIT_CODE
    assert "--unsafe-fixes requires --fix or --diff" in capsys.readouterr().err
