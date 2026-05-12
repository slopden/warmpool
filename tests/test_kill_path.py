"""Regression tests for the SIGTERM / atexit kill path.

Two failure modes these tests catch:

1. Worker SIGTERM handler missing or broken. Linux's default SIGTERM
   disposition terminates the process *without* running ``atexit``, so
   a worker SIGTERM'd by the pool's escalation would skip any
   user-registered ``cleanup`` callable — exactly the case where the
   pool was supposed to release a process-level resource before death.

2. Two-strikes detection missing. A systemic worker-startup wedge
   (driver lock leak, deadlocked import, OS resource exhaustion) must
   surface as :class:`ProcessPoolExhausted` instead of looping forever
   spinning up hung workers.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest
from warmpool import WarmPool
from warmpool._exceptions import ProcessPoolExhausted

from ._helpers import get_pid, sleep_forever, write_marker


def _wait_for_marker(marker_path: Path, timeout: float = 5.0) -> bool:
    """Wait up to *timeout* seconds for *marker_path* to exist.

    Parameters
    ----------
    marker_path
        File whose existence indicates the cleanup hook ran.
    timeout
        Maximum seconds to wait before returning ``False``.

    Returns
    -------
    bool
        ``True`` if the marker appeared within the deadline.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker_path.exists():
            return True
        time.sleep(0.05)
    return False


def test_cleanup_runs_on_sigterm(tmp_path, monkeypatch):
    """Direct SIGTERM to an idle worker must invoke atexit cleanup.

    Without the worker's SIGTERM handler the kernel terminates the
    process before Python can run atexit — the marker file would not
    be written, and any user-registered ``cleanup`` callable would be
    silently skipped.

    We assert via marker-file existence, not by checking whether the
    pid is gone: ``os.kill(pid, 0)`` succeeds against zombies, so a
    worker that has called ``os._exit`` but hasn't been reaped yet
    would falsely look alive.
    """
    marker_path = tmp_path / "cleanup.marker"
    monkeypatch.setenv("WARMPOOL_TEST_MARKER", str(marker_path))

    pool = WarmPool(max_tasks=10, keep_spare=False, cleanup=write_marker)
    try:
        worker_pid = pool.run(get_pid, 5.0)
        assert isinstance(worker_pid, int)
        assert not marker_path.exists()

        os.kill(worker_pid, signal.SIGTERM)

        assert _wait_for_marker(marker_path), (
            "cleanup did not run on SIGTERM — the worker's atexit hooks "
            "were skipped, which would leak any user-registered cleanup"
        )
        assert marker_path.read_text() == "ran"
    finally:
        pool.shutdown()


def test_cleanup_runs_on_timeout_kill(tmp_path, monkeypatch):
    """The pool's timeout path (background SIGTERM → atexit) runs cleanup.

    A worker hangs in ``time.sleep`` and misses its timeout. The pool
    raises ``TimeoutError`` and kicks off the kill in a daemon thread;
    that thread sends SIGTERM, the worker's ``_sigterm_handler`` fires,
    atexit runs, and the marker file is written. The kill is async, so
    poll for the marker rather than expecting it on return.
    """
    marker_path = tmp_path / "cleanup.marker"
    monkeypatch.setenv("WARMPOOL_TEST_MARKER", str(marker_path))

    pool = WarmPool(max_tasks=10, keep_spare=False, cleanup=write_marker)
    try:
        with pytest.raises(TimeoutError):
            pool.run(sleep_forever, timeout=0.3)
        assert _wait_for_marker(marker_path, timeout=3.0), (
            "cleanup did not run on the timeout-driven kill — the pool's "
            "background SIGTERM either never fired or skipped atexit"
        )
        assert marker_path.read_text() == "ran"
    finally:
        pool.shutdown()


def test_two_strikes_raises_pool_exhausted(monkeypatch):
    """Two consecutive ready-signal timeouts → :class:`ProcessPoolExhausted`.

    Without two-strikes detection the pool would respawn workers
    forever against whatever was wedging startup. The constructor's
    retry loop sleeps between attempts; patch that out to keep the
    test fast.
    """
    monkeypatch.setattr("warmpool.pool.time.sleep", lambda *_: None)

    with pytest.raises(ProcessPoolExhausted, match="consecutive worker startups"):
        WarmPool(
            warming=sleep_forever,
            ready_timeout=0.3,
            init_retries=1,
            keep_spare=False,
        )
