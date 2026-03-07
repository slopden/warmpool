"""Pipe-based logging for worker subprocesses.

Provides :class:`PipeHandler` (installed in workers) and
:func:`forward_subprocess_log` (called in the parent) so that log
records cross the process boundary as structured dicts.
"""

from __future__ import annotations

import logging
import traceback
from multiprocessing.connection import Connection


class PipeHandler(logging.Handler):
    """Logging handler that serializes records and sends them through a
    multiprocessing pipe as structured dicts (JSON-ready).

    Installed in the **worker** subprocess so that all log output is
    forwarded to the parent process over the pipe.

    Parameters
    ----------
    conn
        The child-side :class:`~multiprocessing.connection.Connection`.
    """

    def __init__(self, conn: Connection):
        super().__init__()
        self.conn = conn

    def emit(self, record: logging.LogRecord) -> None:
        """Serialize *record* to a dict and send it over the pipe.

        Parameters
        ----------
        record
            The log record to forward.
        """
        try:
            entry: dict = {
                "timestamp": record.created,
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "process_id": record.process,
            }
            if record.exc_info and record.exc_info[1] is not None:
                entry["exception"] = "".join(
                    traceback.format_exception(*record.exc_info)
                )
            self.conn.send(("log", entry, {}))
        except Exception:
            pass


def forward_subprocess_log(
    payload: dict,
    logger: logging.Logger | None = None,
) -> None:
    """Re-emit a subprocess log record via the parent's logging system.

    Parameters
    ----------
    payload
        The structured dict received from :class:`PipeHandler`.
    logger
        Logger to emit on.  Defaults to ``warmpool.subprocess``.
    """
    if logger is None:
        logger = logging.getLogger("warmpool.subprocess")
    level = getattr(logging, payload.get("level", "INFO"), logging.INFO)
    msg = payload.get("message", "")
    extra = {
        k: v
        for k, v in payload.items()
        if k not in ("level", "message", "levelname", "levelno")
    }
    logger.log(level, msg, extra=extra)
