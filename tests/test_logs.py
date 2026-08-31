"""Reports go to stdout unformatted; diagnostics go to stderr and can be silenced."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from fraudsim import logs

ROOT = Path(__file__).resolve().parent.parent


def run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter, so handler wiring is exercised from cold."""
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )


SPLIT = """
from fraudsim.logs import configure, get_logger, emit
import sys
configure(level=sys.argv[1] if len(sys.argv) > 1 else None)
get_logger("fraudsim.probe").info("DIAGNOSTIC")
emit("REPORT")
"""


def test_report_goes_to_stdout_and_diagnostics_to_stderr() -> None:
    done = run(SPLIT)
    assert done.stdout == "REPORT\n"
    assert "DIAGNOSTIC" in done.stderr
    assert "REPORT" not in done.stderr


def test_report_carries_no_prefix() -> None:
    """A rendered table must not gain a timestamp on every row."""
    done = run(SPLIT)
    assert done.stdout.strip() == "REPORT"


def test_diagnostics_carry_level_and_logger_name() -> None:
    done = run(SPLIT)
    assert "INFO" in done.stderr
    assert "fraudsim.probe" in done.stderr


def test_raising_the_level_silences_diagnostics_but_not_the_report() -> None:
    done = run(SPLIT, "WARNING")
    assert done.stdout == "REPORT\n"
    assert done.stderr == ""


def test_env_var_sets_the_level() -> None:
    import os
    env = {**os.environ, "GAUNTLET_LOG_LEVEL": "WARNING"}
    done = subprocess.run([sys.executable, "-c", SPLIT], capture_output=True,
                          text=True, cwd=ROOT, env=env, check=False)
    assert done.stdout == "REPORT\n"
    assert done.stderr == ""


def test_emit_with_no_argument_is_a_blank_line() -> None:
    """It replaces a bare print(), which the report layouts rely on."""
    done = run("from fraudsim.logs import emit\nemit()\nemit('x')\n")
    assert done.stdout == "\nx\n"


def test_handlers_follow_a_replaced_stream(capsys: pytest.CaptureFixture[str]) -> None:
    """dictConfig binds the stream once; the handlers must resolve it per write.

    Without this, pytest's capsys and any redirect installed after configure()
    would be written past rather than to.
    """
    logs.configure()
    logs.emit("CAPTURED")
    assert "CAPTURED" in capsys.readouterr().out


def test_the_report_logger_does_not_propagate() -> None:
    """Otherwise every report row would also appear on stderr, timestamped."""
    logs.configure()
    assert logging.getLogger(logs.REPORT_LOGGER).propagate is False


def test_configuration_falls_back_when_the_yaml_is_absent(tmp_path: Path) -> None:
    """An installed copy with no configs/ tree still logs."""
    logs.configure(path=tmp_path / "missing.yaml")
    assert logging.getLogger(logs.REPORT_LOGGER).handlers
