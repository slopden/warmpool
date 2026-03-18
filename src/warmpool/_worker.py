"""Worker subprocess entry point.

This module is imported by the spawned child process.  It sets up
pipe-based logging, runs the optional warming callable, then enters a
receive-execute-send loop until the parent sends a shutdown sentinel
(``func is None``) or the pipe breaks.
"""

from __future__ import annotations

import logging
import time
from multiprocessing.connection import Connection
from typing import Callable

from ._logging import PipeHandler


def _worker_process(
    connection: Connection,
    log_level: int = logging.DEBUG,
    warming: Callable | None = None,
) -> None:
    """Entry point for the worker subprocess.

    Parameters
    ----------
    connection
        Child-side pipe connection shared with the parent.
    warming
        Optional callable invoked once on startup (e.g. to pre-import
        modules).  Its return value is sent to the parent.

    Notes
    -----
    1. Replaces all root-logger handlers with a :class:`PipeHandler` so
       every log record is forwarded to the parent as a structured dict.
    2. Calls *warming* if provided.
    3. Sends a ``("ready", init_result, {})`` message, then enters the task loop.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(PipeHandler(connection))
    root.setLevel(log_level)

    init_result = warming() if warming is not None else None
    connection.send(("ready", init_result, {}))

    try:
        while True:
            if not connection.poll(timeout=None):
                continue

            try:
                function, args, kwargs = connection.recv()
                if function is None:  # shutdown sentinel
                    break

                start = time.perf_counter()
                result = function(*args, **kwargs)
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                connection.send(("success", result, {"elapsed_ms": elapsed_ms}))
            except Exception as error:
                # Guard against unpicklable exceptions (common with
                # C-API wrappers).  If the exception can't be pickled
                # the parent would see a silent worker death instead
                # of a useful error message.
                try:
                    connection.send(("error", error, {}))
                except Exception:
                    connection.send(("error", RuntimeError(repr(error)), {}))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()
