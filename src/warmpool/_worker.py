"""Worker subprocess entry point.

This module is imported by the spawned child process.  It sets up
pipe-based logging, pre-imports warm modules, then enters a
receive-execute-send loop until the parent sends a shutdown sentinel
(``func is None``) or the pipe breaks.
"""

from __future__ import annotations

import logging
import time
from importlib import import_module
from multiprocessing.connection import Connection
from typing import Callable

from ._logging import PipeHandler


def _worker_process(
    conn: Connection,
    warm_modules: list[str],
    log_level: int = logging.DEBUG,
    init_func: Callable | None = None,
) -> None:
    """Entry point for the worker subprocess.

    Parameters
    ----------
    conn
        Child-side pipe connection shared with the parent.
    warm_modules
        Module names to pre-import so startup cost is amortized.

    Notes
    -----
    1. Replaces all root-logger handlers with a :class:`PipeHandler` so
       every log record is forwarded to the parent as a structured dict.
    2. Pre-imports *warm_modules*.
    3. Sends a ``("ready", None, {})`` message, then enters the task loop.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(PipeHandler(conn))
    root.setLevel(log_level)

    for name in warm_modules:
        try:
            import_module(name)
        except ImportError:
            root.warning(f"Failed to warm module: {name}")

    if init_func is not None:
        init_func()

    conn.send(("ready", None, {}))

    try:
        while True:
            if not conn.poll(timeout=None):
                continue

            try:
                func, args, kwargs = conn.recv()
                if func is None:  # shutdown sentinel
                    break

                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                conn.send(("success", result, {"elapsed_ms": elapsed_ms}))
            except Exception as e:
                # Guard against unpicklable exceptions (common with
                # C-API wrappers).  If the exception can't be pickled
                # the parent would see a silent worker death instead
                # of a useful error message.
                try:
                    conn.send(("error", e, {}))
                except Exception:
                    conn.send(("error", RuntimeError(repr(e)), {}))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        conn.close()
