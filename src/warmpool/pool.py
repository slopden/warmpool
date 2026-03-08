"""Single-worker subprocess pool with hard-kill timeouts.

Runs callables in a spawned subprocess that can be SIGKILLed when
C-API code (OpenCASCADE, etc.) ignores Python signals.  An optional
spare worker is pre-warmed in the background so that rotation after
task-limit exhaustion or crash is near-instant.
"""

from __future__ import annotations

import asyncio
import atexit
import enum
import logging
import multiprocessing as mp
import time
import weakref
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, NoReturn

import psutil

from ._exceptions import ProcessPoolExhausted
from ._logging import forward_subprocess_log
from ._worker import _worker_process

log = logging.getLogger(__name__)

# Seconds to wait when polling a pipe for data.
_POLL_TIMEOUT = 0.1
# Seconds to wait for a worker to join after a graceful shutdown signal.
_JOIN_TIMEOUT = 0.5
# Seconds to wait for a process tree to die after SIGKILL.
_KILL_WAIT = 1.0

_active_pools: weakref.WeakSet[PoolWithTimeout] = weakref.WeakSet()


def _cleanup_all_pools() -> None:
    """Shut down every live pool at interpreter exit."""
    for pool in list(_active_pools):
        try:
            pool.shutdown()
        except Exception:
            pass


atexit.register(_cleanup_all_pools)


class PoolStatus(enum.Enum):
    """The pool's readiness state.

    Returned by :attr:`PoolWithTimeout.status`.  Every decision point
    in the pool dispatches on this enum with :func:`_assert_never` in
    the ``else`` branch so that mypy proves exhaustive handling.

    Attributes
    ----------
    READY
        Active worker is alive and under the task limit.
    NEEDS_ROTATION
        Active worker is spent or dead, but a spare can take over.
    EXHAUSTED
        No workers available and no spare to promote.
    SHUTDOWN
        The pool has been explicitly shut down.
    """

    READY = "ready"
    NEEDS_ROTATION = "rotation"
    EXHAUSTED = "exhausted"
    SHUTDOWN = "shutdown"


def _assert_never(value: NoReturn) -> NoReturn:
    """Statically assert all enum cases are handled.

    Mypy narrows the type through each ``if``/``elif`` branch.  If all
    :class:`PoolStatus` members are covered, the remaining type at the
    ``else`` is ``Never``/``NoReturn`` and this call type-checks.
    Adding a new enum member without a matching branch causes a mypy
    error.
    """
    raise AssertionError(f"Unhandled status: {value!r}")


@dataclass
class WorkerHandle:
    """Bookkeeping for a single worker subprocess.

    Parameters
    ----------
    process
        The :class:`multiprocessing.Process` instance.
    conn
        Parent-side pipe connection.
    child_conn
        Child-side pipe connection (kept open so we can close it on
        cleanup).
    ready
        ``True`` once the worker has sent its ``"ready"`` message.
    task_count
        Number of tasks dispatched to this worker.
    last_metadata
        Metadata dict from the most recent completed task.
    """

    process: mp.Process
    conn: Connection
    child_conn: Connection
    ready: bool = False
    task_count: int = 0
    last_metadata: dict[str, Any] = field(default_factory=dict)
    init_result: Any = None


class PoolWithTimeout:
    """Single-worker subprocess pool with hard-kill timeouts.

    Runs functions in a spawned subprocess that can be SIGKILLed when
    C-API code (OpenCASCADE, etc.) ignores Python signals.

    .. note::
        This class is **not** thread-safe.  Do not call :meth:`run` from
        multiple threads concurrently.

    Parameters
    ----------
    warm_modules
        Module names to pre-import in the worker on startup.
    max_tasks
        Maximum tasks a single worker may handle before rotation.
    keep_spare
        If ``True``, a spare worker is pre-warmed in the background so
        rotation is near-instant.
    ready_timeout
        Seconds to wait for a worker to send its ``"ready"`` signal.
    max_memory
        Maximum RSS in bytes before the worker is rotated.
    max_memory_percent
        Maximum RSS as a fraction of total system memory (0.0–1.0)
        before the worker is rotated.
    """

    def __init__(
        self,
        warm_modules: list[str] | None = None,
        max_tasks: int = 50,
        keep_spare: bool = True,
        ready_timeout: float = 30.0,
        max_memory: int | None = None,
        max_memory_percent: float | None = None,
        init_func: Callable | None = None,
    ) -> None:
        self._warm_modules = warm_modules or []
        self._max_tasks = max_tasks
        self._keep_spare = keep_spare
        self._ready_timeout = ready_timeout
        self._init_func = init_func
        self._max_memory = max_memory
        # Pre-compute absolute byte limit from percentage (avoid per-task psutil call).
        if max_memory_percent is not None:
            clamped = max(0.0, min(1.0, max_memory_percent))
            self._max_memory_percent_bytes: int | None = int(
                clamped * psutil.virtual_memory().total
            )
        else:
            self._max_memory_percent_bytes = None

        self._active: WorkerHandle | None = None
        self._spare: WorkerHandle | None = None
        self._shutdown = False
        # Pool-level cache so elapsed_ms survives rotation.
        self._last_elapsed_ms: int | None = None
        self._last_memory_rss: int | None = None

        _active_pools.add(self)

        # Start primary worker (blocking).
        self._active = self._start_worker(block_ready=True)

        # Start spare (non-blocking) if requested.
        if self._keep_spare:
            self._spare = self._start_worker(block_ready=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def status(self) -> PoolStatus:
        """The pool's current readiness state.

        Returns
        -------
        PoolStatus
            Pure query — no side effects, no mutations.
        """
        if self._shutdown:
            return PoolStatus.SHUTDOWN
        if (
            self._active is not None
            and self._active.process.is_alive()
            and self._active.task_count < self._max_tasks
        ):
            return PoolStatus.READY
        if self._keep_spare:
            return PoolStatus.NEEDS_ROTATION
        return PoolStatus.EXHAUSTED

    @property
    def init_result(self) -> Any:
        """Return value of ``init_func`` from the active worker, or ``None``."""
        return self._active.init_result if self._active else None

    @property
    def last_elapsed_ms(self) -> int | None:
        """Wall-clock milliseconds the last completed task took.

        Returns
        -------
        int or None
            ``None`` if no task has completed yet.
        """
        return self._last_elapsed_ms

    @property
    def last_memory_rss(self) -> int | None:
        """RSS in bytes of the worker after the last completed task.

        Returns ``None`` if no task has completed or memory checking is disabled.
        """
        return self._last_memory_rss

    def run(self, func: Callable, timeout: float, **kwargs: Any) -> Any:
        """Run *func* in the worker subprocess (blocking).

        Parameters
        ----------
        func
            A picklable callable to execute in the worker.
        timeout
            Hard timeout in seconds; the worker is SIGKILLed if it
            exceeds this.
        **kwargs
            Keyword arguments forwarded to *func*.

        Returns
        -------
        Any
            Whatever *func* returns.

        Raises
        ------
        TimeoutError
            If the worker exceeds *timeout*.
        ProcessPoolExhausted
            If the pool has no available workers.
        """
        status = self.status
        if status is PoolStatus.READY:
            pass
        elif status is PoolStatus.NEEDS_ROTATION:
            self._rotate_worker()
        elif status is PoolStatus.EXHAUSTED:
            # Capture diagnostics *before* cleanup.
            task_count = self._active.task_count if self._active else 0
            exit_code = self._active.process.exitcode if self._active else None
            if self._active is not None:
                self._shutdown_worker(self._active)
                self._active = None
            raise ProcessPoolExhausted(
                f"tasks={task_count}/{self._max_tasks}",
                exit_code=exit_code,
            )
        elif status is PoolStatus.SHUTDOWN:
            raise ProcessPoolExhausted("Pool is shut down")
        else:
            _assert_never(status)

        # At this point self._active is guaranteed non-None.
        handle = self._active
        assert handle is not None  # narrowing for mypy

        # Send the task.
        handle.conn.send((func, (), kwargs))
        handle.task_count += 1

        # Wait for result.
        try:
            result = self._wait_for_result(handle, func, timeout)
        except (TimeoutError, ProcessPoolExhausted):
            self._kill_worker(handle)
            self._active = None
            if self._keep_spare:
                try:
                    self._promote_spare()
                except Exception:
                    log.warning("Failed to promote spare after error", exc_info=True)
            raise

        # Persist elapsed_ms at pool level so it survives rotation.
        self._last_elapsed_ms = handle.last_metadata.get("elapsed_ms")

        # Rotate after the final allowed task or memory limit exceeded.
        memory_exceeded = self._exceeds_memory_limit(handle)
        if handle.task_count >= self._max_tasks or memory_exceeded:
            self._shutdown_worker(handle)
            self._active = None
            if self._keep_spare:
                try:
                    self._promote_spare()
                except Exception:
                    log.warning("Failed to promote spare after rotation", exc_info=True)

        return result

    async def arun(self, func: Callable, timeout: float, **kwargs: Any) -> Any:
        """Async wrapper around :meth:`run`.

        Parameters
        ----------
        func
            A picklable callable.
        timeout
            Hard timeout in seconds.
        **kwargs
            Forwarded to *func*.

        Returns
        -------
        Any
            Whatever *func* returns.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.run(func, timeout, **kwargs)
        )

    def shutdown(self) -> None:
        """Shut down all workers and mark the pool as dead."""
        self._shutdown = True
        if self._active is not None:
            self._shutdown_worker(self._active)
            self._active = None
        if self._spare is not None:
            self._shutdown_worker(self._spare)
            self._spare = None

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(self, block_ready: bool = True) -> WorkerHandle:
        """Spawn a new worker subprocess.

        Parameters
        ----------
        block_ready
            If ``True``, block until the worker sends ``"ready"``.

        Returns
        -------
        WorkerHandle
        """
        parent_conn, child_conn = Pipe()
        log_level = logging.getLogger().getEffectiveLevel()
        try:
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=_worker_process,
                args=(child_conn, self._warm_modules, log_level, self._init_func),
            )
        except RuntimeError:
            proc = Process(
                target=_worker_process,
                args=(child_conn, self._warm_modules, log_level, self._init_func),
            )

        proc.start()
        handle = WorkerHandle(process=proc, conn=parent_conn, child_conn=child_conn)
        log.info(f"Started worker pid={proc.pid}")

        if block_ready:
            if not self._wait_for_ready(handle):
                self._kill_worker(handle)
                raise RuntimeError(
                    f"Worker pid={handle.process.pid} failed to become ready "
                    f"within {self._ready_timeout}s"
                )

        return handle

    def _wait_for_ready(self, handle: WorkerHandle) -> bool:
        """Block until *handle* sends ``"ready"`` or the timeout expires.

        Parameters
        ----------
        handle
            The worker to wait on.

        Returns
        -------
        bool
            ``True`` if the worker became ready, ``False`` on timeout or error.
        """
        deadline = time.perf_counter() + self._ready_timeout
        while time.perf_counter() < deadline:
            remaining = max(0.01, deadline - time.perf_counter())
            if not handle.conn.poll(timeout=remaining):
                break
            try:
                status, payload, _ = handle.conn.recv()
                if status == "log":
                    forward_subprocess_log(payload)
                    continue
                if status == "ready":
                    handle.ready = True
                    handle.init_result = payload
                    return True
                log.warning(f"Expected 'ready', got: {status}")
                return False
            except Exception as e:
                log.warning(f"Failed to receive ready signal: {e}")
                return False
        log.warning("Worker didn't send ready signal within timeout")
        return False

    def _wait_for_result(
        self, handle: WorkerHandle, func: Callable, timeout: float
    ) -> Any:
        """Poll *handle* for the task result, forwarding log messages.

        Parameters
        ----------
        handle
            The active worker handle.
        func
            The function that was dispatched (used for error messages).
        timeout
            Hard timeout in seconds.

        Returns
        -------
        Any
            The return value of *func*.

        Raises
        ------
        TimeoutError
            If *timeout* is exceeded.
        ProcessPoolExhausted
            If the worker dies mid-task.
        """
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            if handle.conn.poll(timeout=_POLL_TIMEOUT):
                try:
                    status, payload, metadata = handle.conn.recv()
                except (EOFError, BrokenPipeError):
                    ec = handle.process.exitcode
                    raise ProcessPoolExhausted(
                        f"Subprocess died during `{func.__name__}`",
                        exit_code=ec,
                    )

                if status == "log":
                    forward_subprocess_log(payload)
                    continue

                handle.last_metadata = metadata

                if status == "success":
                    return payload
                if status == "error":
                    raise payload

            if not handle.process.is_alive():
                ec = handle.process.exitcode
                raise ProcessPoolExhausted(
                    f"Subprocess died during `{func.__name__}`",
                    exit_code=ec,
                )

        raise TimeoutError(f"`{func.__name__}` timed out after {timeout}s")

    def _exceeds_memory_limit(self, handle: WorkerHandle) -> bool:
        if self._max_memory is None and self._max_memory_percent_bytes is None:
            return False
        try:
            rss = psutil.Process(handle.process.pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            return False
        self._last_memory_rss = rss
        if self._max_memory is not None and rss > self._max_memory:
            log.info(
                f"Worker pid={handle.process.pid} RSS {rss:,}B exceeds max_memory={self._max_memory:,}B, rotating"
            )
            return True
        if (
            self._max_memory_percent_bytes is not None
            and rss > self._max_memory_percent_bytes
        ):
            log.info(
                f"Worker pid={handle.process.pid} RSS {rss:,}B exceeds percent limit ({self._max_memory_percent_bytes:,}B), rotating"
            )
            return True
        return False

    def _rotate_worker(self) -> None:
        """Shut down the spent active worker and promote the spare.

        Called when :attr:`status` is :attr:`PoolStatus.NEEDS_ROTATION`.
        """
        if self._active is not None:
            self._shutdown_worker(self._active)
            self._active = None
        self._promote_spare()

    def _promote_spare(self) -> None:
        """Make the spare worker active and replenish the spare.

        If the spare has died or never became ready (e.g. deadlocked
        during warm-module import), it is killed and a fresh worker is
        cold-started instead.
        """
        if self._spare is not None:
            if self._spare.process.is_alive():
                if not self._spare.ready:
                    self._wait_for_ready(self._spare)
                if self._spare.ready:
                    self._active = self._spare
                    self._spare = None
                else:
                    # Spare is alive but never became ready — kill it.
                    log.warning(
                        f"Spare pid={self._spare.process.pid} never became ready, killing"
                    )
                    self._kill_worker(self._spare)
                    self._spare = None
                    self._active = self._start_worker(block_ready=True)
            else:
                # Spare died — clean up and cold-start.
                self._close_worker(self._spare)
                self._spare = None
                self._active = self._start_worker(block_ready=True)
        else:
            self._active = self._start_worker(block_ready=True)

        # Replenish spare.
        try:
            self._spare = self._start_worker(block_ready=False)
        except Exception:
            log.warning("Failed to start spare worker", exc_info=True)
            self._spare = None

    def _shutdown_worker(self, handle: WorkerHandle) -> None:
        """Gracefully shut down *handle*; escalate to kill if needed.

        Parameters
        ----------
        handle
            The worker to shut down.
        """
        if handle.process.is_alive():
            try:
                handle.conn.send((None, (), {}))
            except (BrokenPipeError, OSError):
                pass
            handle.process.join(timeout=_JOIN_TIMEOUT)
            if handle.process.is_alive():
                self._kill_worker(handle)
                return
        self._close_worker(handle)

    def _kill_worker(self, handle: WorkerHandle) -> None:
        """SIGTERM then SIGKILL the worker and its entire process tree.

        Parameters
        ----------
        handle
            The worker to kill.
        """
        if not handle.process.is_alive():
            self._close_worker(handle)
            return
        try:
            proc = psutil.Process(handle.process.pid)
            children = proc.children(recursive=True)
            for child in children:
                child.terminate()
            proc.terminate()
            gone, alive = psutil.wait_procs(children + [proc], timeout=0.1)
            for p in alive:
                p.kill()
            psutil.wait_procs(alive, timeout=_KILL_WAIT)
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        except psutil.TimeoutExpired:
            log.warning("Process tree still alive after SIGKILL")
        except Exception:
            log.error("Error killing process tree", exc_info=True)
        self._close_worker(handle)

    def _close_worker(self, handle: WorkerHandle) -> None:
        """Join the process and close both pipe endpoints.

        Parameters
        ----------
        handle
            The worker whose resources should be freed.
        """
        handle.process.join(timeout=_POLL_TIMEOUT)
        try:
            handle.conn.close()
        except Exception:
            pass
        try:
            handle.child_conn.close()
        except Exception:
            pass
