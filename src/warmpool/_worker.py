"""Worker subprocess entry point.

This module is imported by the spawned child process.  It sets up
pipe-based logging, runs the optional warming callable, then enters a
receive-execute-send loop until the parent sends a shutdown sentinel
(``func is None``) or the pipe breaks.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import time
from collections.abc import Callable
from multiprocessing.connection import Connection

from ._logging import PipeHandler


def _sigterm_handler(_signum: int, _frame: object) -> None:
    """Run ``atexit`` hooks then exit immediately on SIGTERM.

    Linux's default SIGTERM disposition terminates the process
    *without* running ``atexit``, which would skip any user-registered
    ``cleanup`` callable — exactly the path that leaves external
    resources (device handles, locks, queues) half-released.

    ``sys.exit(0)`` is not a reliable substitute: when the worker is
    blocked in ``connection.poll(timeout=None)`` (``epoll_wait``), the
    ``SystemExit`` raised from this handler does not reliably unwind
    out of that C frame on every Python version. Running ``atexit``
    hooks explicitly and then ``os._exit`` gives the same observable
    behavior with no dependency on the interpreter's
    signal-out-of-C-call machinery.
    """
    atexit._run_exitfuncs()
    os._exit(0)


def _worker_process(
    connection: Connection,
    log_level: int = logging.DEBUG,
    warming: Callable | None = None,
    cleanup: Callable | None = None,
) -> None:
    """Entry point for the worker subprocess.

    Parameters
    ----------
    connection
        Child-side pipe connection shared with the parent.
    warming
        Optional callable invoked once on startup (e.g. to pre-import
        modules).  Its return value is sent to the parent.
    cleanup
        Optional callable registered via ``atexit`` so it runs on
        SIGTERM exit (before Python module teardown). Use for
        releasing process-level resources that would otherwise leak
        when the OS reaps the worker.

    Notes
    -----
    1. Replaces all root-logger handlers with a :class:`PipeHandler` so
       every log record is forwarded to the parent as a structured dict.
    2. Installs a SIGTERM handler that runs ``atexit`` then ``os._exit``,
       so cleanup hooks fire even when the pool escalates to a kill.
    3. Registers *cleanup* with ``atexit`` before *warming* runs — so
       cleanup still fires if *warming* itself errors out.
    4. Calls *warming* if provided.
    5. Sends a ``("ready", init_result, {})`` message, then enters the task loop.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(PipeHandler(connection))
    root.setLevel(log_level)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    if cleanup is not None:
        atexit.register(cleanup)

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
