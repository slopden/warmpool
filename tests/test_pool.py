import os

import pytest

from warmpool import PoolStatus, WarmPool, ProcessPoolExhausted

from ._helpers import (
    add,
    check_module_imported,
    force_exit,
    get_pid,
    raise_unpicklable,
    raise_value_error,
    scipy_eigh_huge,
    sleep_forever,
    warm_json,
    warm_numpy_scipy,
)


def test_run_returns_result(pool):
    assert pool.run(add, 5.0, a=2, b=3) == 5


def test_run_with_kwargs(pool):
    assert pool.run(add, 5.0, a=10, b=20) == 30


def test_subprocess_isolation(pool):
    child_pid = pool.run(get_pid, 5.0)
    assert isinstance(child_pid, int)
    assert child_pid != os.getpid()


def test_timeout_raises(pool):
    with pytest.raises(TimeoutError, match="sleep_forever"):
        pool.run(sleep_forever, 0.5)


def test_hard_kill_verified(pool):
    child_pid = pool.run(get_pid, 5.0)
    assert isinstance(child_pid, int)
    with pytest.raises(TimeoutError, match="sleep_forever"):
        pool.run(sleep_forever, 0.5)
    # Worker should be dead
    with pytest.raises(OSError):
        os.kill(child_pid, 0)


def test_error_propagation(pool):
    with pytest.raises(ValueError, match="boom"):
        pool.run(raise_value_error, 5.0, message="boom")


def test_exhaustion_without_spare():
    pool = WarmPool(max_tasks=2, keep_spare=False)
    try:
        assert pool.run(add, 5.0, a=1, b=1) == 2
        assert pool.run(add, 5.0, a=2, b=2) == 4
        assert pool.status is PoolStatus.EXHAUSTED
        with pytest.raises(ProcessPoolExhausted):
            pool.run(add, 5.0, a=3, b=3)
    finally:
        pool.shutdown()


def test_run_after_shutdown(pool):
    pool.shutdown()
    assert pool.status is PoolStatus.SHUTDOWN
    with pytest.raises(ProcessPoolExhausted):
        pool.run(add, 5.0, a=1, b=1)


def test_double_shutdown(pool):
    pool.shutdown()
    pool.shutdown()  # should not raise
    assert pool.status is PoolStatus.SHUTDOWN


def test_multiple_sequential_tasks(pool):
    for i in range(5):
        assert pool.run(add, 5.0, a=i, b=i) == i * 2


def test_elapsed_ms(pool):
    assert pool.run(add, 5.0, a=1, b=2) == 3
    assert isinstance(pool.last_elapsed_ms, int)
    assert pool.last_elapsed_ms >= 0


def test_warming():
    pool = WarmPool(warming=warm_json)
    try:
        assert pool.run(check_module_imported, 5.0, module_name="json") is True
    finally:
        pool.shutdown()


def test_worker_crash(pool):
    with pytest.raises(ProcessPoolExhausted) as exc_info:
        pool.run(force_exit, 5.0, code=1)
    assert exc_info.value.exit_code == 1


@pytest.mark.asyncio
async def test_arun():
    pool = WarmPool()
    try:
        result = await pool.arun(add, 5.0, a=1, b=2)
        assert result == 3
    finally:
        pool.shutdown()


def test_unpicklable_exception(pool):
    """Unpicklable exceptions are wrapped in RuntimeError, not lost."""
    with pytest.raises(RuntimeError, match="can't be pickled"):
        pool.run(raise_unpicklable, 5.0)


def test_kill_c_extension_blocked_worker():
    """SIGKILL works even when the worker is stuck inside C code (LAPACK).

    ``scipy.linalg.eigh`` on a 5000x5000 matrix runs for ~10s+ entirely
    in Fortran/C with the GIL held, so Python signal handlers never fire.
    The pool's hard-kill (SIGKILL via psutil) is the only way to stop it.
    """
    pool = WarmPool(
        warming=warm_numpy_scipy,
        keep_spare=True,
    )
    try:
        with pytest.raises(TimeoutError, match="scipy_eigh_huge"):
            # 0.5s timeout — LAPACK will be deep in C code by then
            pool.run(scipy_eigh_huge, timeout=0.5, n=5000)

        # Pool recovered via spare — prove it still works.
        assert pool.run(add, 5.0, a=7, b=8) == 15
    finally:
        pool.shutdown()
