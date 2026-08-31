"""Two output channels: diagnostics on stderr, rendered reports on stdout.

The package had no logging at all. 186 print calls went to stdout, mixing a
run's progress in with the report it produced, so `... metrics > out.txt`
captured the progress lines too and a slow phase could not be watched while its
output was redirected. Three separate ad-hoc progress mechanisms had grown up to
work around it.

Reports keep a bare formatter, because a report is a rendered table and a
timestamp prefix in front of every row would ruin it. Diagnostics carry the
usual level and logger name, and go to stderr, so the two never interleave in a
pipe.

    from .logs import get_logger, emit

    log = get_logger(__name__)
    log.info("fitting the defender")   # stderr, silenced by --log-level WARNING
    emit(metrics.render())             # stdout, always, unformatted
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import TextIO

import yaml

from .paths import DEFAULT_LOGGING

REPORT_LOGGER = "fraudsim.report"

_configured = False


class StdoutHandler(logging.StreamHandler[TextIO]):
    """A stream handler that looks up sys.stdout on every write.

    logging.config binds `ext://sys.stdout` once, at configure time. Anything
    that replaces the stream afterwards -- pytest's capsys, a redirect inside a
    context manager -- would then be written past rather than to. Resolving on
    each emit keeps the handler pointed at whatever stdout currently is.

    StreamHandler.__init__ assigns self.stream, so the live lookup cannot be a
    read-only property. It is a normal attribute that emit() refreshes.
    """

    stream_name = "stdout"

    def __init__(self) -> None:
        super().__init__(getattr(sys, self.stream_name))

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = getattr(sys, self.stream_name)
        super().emit(record)


class StderrHandler(StdoutHandler):
    """The diagnostic channel, resolved live for the same reason."""

    stream_name = "stderr"


def configure(path: Path | None = None, level: str | None = None) -> None:
    """Apply the logging configuration once per process.

    The level resolves from the argument, then GAUNTLET_LOG_LEVEL, then whatever
    the file says, so a caller can raise or lower it without editing the file.
    """
    global _configured
    if _configured and level is None:
        return

    config_path = path or DEFAULT_LOGGING
    if config_path.is_file():
        config: dict[str, object] = yaml.safe_load(config_path.read_text())
    else:
        config = _fallback()
    logging.config.dictConfig(config)

    chosen = level or os.environ.get("GAUNTLET_LOG_LEVEL")
    if chosen:
        logging.getLogger("fraudsim").setLevel(chosen.upper())
    _configured = True


def _fallback() -> dict[str, object]:
    """Enough configuration to run from an installed copy with no configs/ tree."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "diagnostic": {"format": "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                           "datefmt": "%H:%M:%S"},
            "bare": {"format": "%(message)s"},
        },
        "handlers": {
            "console": {"class": "fraudsim.logs.StderrHandler",
                        "formatter": "diagnostic", "level": "INFO"},
            "report": {"class": "fraudsim.logs.StdoutHandler",
                       "formatter": "bare", "level": "INFO"},
        },
        "loggers": {REPORT_LOGGER: {"handlers": ["report"], "level": "INFO",
                                    "propagate": False}},
        "root": {"handlers": ["console"], "level": "INFO"},
    }


def get_logger(name: str) -> logging.Logger:
    """A diagnostic logger for one module, writing to stderr."""
    if not _configured:
        configure()
    return logging.getLogger(name)


def emit(message: str = "") -> None:
    """One line of rendered report output, on stdout, with no prefix."""
    if not _configured:
        configure()
    logging.getLogger(REPORT_LOGGER).info(message)
