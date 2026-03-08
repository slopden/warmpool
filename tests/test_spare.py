import multiprocessing as mp
import time
from multiprocessing import Pipe

import psutil
import pytest

from warmpool import PoolWithTimeout
from warmpool.pool import WorkerHandle

from ._helpers import add, get_pid, hung_worker_process, sleep_forever


def test_spare_created_on_init(spare_pool):
    assert spare_pool._spare is not None
    assert spare_pool._spare.process.is_alive()


def test_auto_rotation_on_exhaustion():
    p = PoolWithTimeout(max_tasks=2, keep_spare=True)
    try:
        assert p.run(add, 5.0, a=1, b=1) == 2
        pid2 = p.run(get_pid, 5.0)
        assert isinstance(pid2, int)
        # After 2 tasks the active worker should rotate
        pid3 = p.run(get_pid, 5.0)
        assert isinstance(pid3, int)
        assert pid3 != pid2
    finally:
        p.shutdown()


def test_rotation_after_timeout():
    p = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        with pytest.raises(TimeoutError, match="sleep_forever"):
            p.run(sleep_forever, 0.5)
        # Pool should still work via spare
        assert p.run(add, 5.0, a=1, b=2) == 3
    finally:
        p.shutdown()


def test_multiple_rotations():
    p = PoolWithTimeout(max_tasks=1, keep_spare=True)
    try:
        for i in range(5):
            assert p.run(add, 5.0, a=i, b=1) == i + 1
    finally:
        p.shutdown()


def test_spare_replenished():
    p = PoolWithTimeout(max_tasks=1, keep_spare=True)
    try:
        assert p.run(add, 5.0, a=1, b=1) == 2
        # After rotation, spare should be replenished
        assert p._spare is not None
        assert p._spare.process.is_alive()
    finally:
        p.shutdown()


def test_rotation_faster_than_cold_start():
    # Time a cold-start pool creation
    t0 = time.perf_counter()
    cold = PoolWithTimeout(warm_modules=["json", "xml"], keep_spare=False)
    cold_time = time.perf_counter() - t0
    cold.shutdown()
    assert cold_time > 0

    # Create a pool with spare already warming
    p = PoolWithTimeout(max_tasks=1, keep_spare=True, warm_modules=["json", "xml"])
    # Run a task to exhaust active and trigger promotion
    assert p.run(add, 5.0, a=1, b=1) == 2
    # Give spare time to be ready
    time.sleep(0.5)
    # Time the promotion
    t0 = time.perf_counter()
    assert p.run(add, 5.0, a=2, b=2) == 4
    spare_time = time.perf_counter() - t0
    p.shutdown()
    assert spare_time > 0

    # Spare promotion should be faster than cold start
    assert spare_time < cold_time


def test_spare_hung_during_import_is_killed():
    """Spare that never sends 'ready' is killed on promotion and replaced
    with a cold-started worker.
    """
    p = PoolWithTimeout(max_tasks=1, keep_spare=True, ready_timeout=0.2)
    try:
        # Replace the spare with a process that never sends "ready".
        p._shutdown_worker(p._spare)
        parent_conn, child_conn = Pipe()
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=hung_worker_process, args=(child_conn, [], 10))
        proc.start()
        hung_pid = proc.pid
        p._spare = WorkerHandle(
            process=proc, conn=parent_conn, child_conn=child_conn, ready=False
        )

        # Exhaust the active worker to trigger rotation → promotion.
        assert p.run(add, 5.0, a=1, b=1) == 2

        # The hung spare should have been killed (not promoted).
        assert p._active is not None
        assert p._active.process.pid != hung_pid
        assert p._active.ready

        # Verify the hung process was actually killed.
        assert (
            not psutil.pid_exists(hung_pid)
            or psutil.Process(hung_pid).status() == psutil.STATUS_ZOMBIE
        ), f"Hung process {hung_pid} was not killed"

        # Pool should still work after recovering.
        assert p.run(add, 5.0, a=2, b=2) == 4
    finally:
        p.shutdown()


def test_wait_for_ready_returns_false_on_timeout():
    """_wait_for_ready returns False when the worker never sends 'ready'."""
    p = PoolWithTimeout(max_tasks=1, keep_spare=False, ready_timeout=0.2)
    try:
        parent_conn, child_conn = Pipe()
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=hung_worker_process, args=(child_conn, [], 10))
        proc.start()
        handle = WorkerHandle(
            process=proc, conn=parent_conn, child_conn=child_conn, ready=False
        )

        result = p._wait_for_ready(handle)
        assert result is False
        assert handle.ready is False

        p._kill_worker(handle)
    finally:
        p.shutdown()
