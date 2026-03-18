import multiprocessing
import time
from multiprocessing import Pipe

import psutil
import pytest

from warmpool import WarmPool
from warmpool.pool import WorkerHandle

from ._helpers import add, get_pid, hung_worker_process, sleep_forever, warm_json_xml


def test_spare_created_on_init(spare_pool):
    assert spare_pool._spare is not None
    assert spare_pool._spare.process.is_alive()


def test_auto_rotation_on_exhaustion():
    pool = WarmPool(max_tasks=2, keep_spare=True)
    try:
        assert pool.run(add, 5.0, a=1, b=1) == 2
        pid2 = pool.run(get_pid, 5.0)
        assert isinstance(pid2, int)
        # After 2 tasks the active worker should rotate
        pid3 = pool.run(get_pid, 5.0)
        assert isinstance(pid3, int)
        assert pid3 != pid2
    finally:
        pool.shutdown()


def test_rotation_after_timeout():
    pool = WarmPool(max_tasks=50, keep_spare=True)
    try:
        with pytest.raises(TimeoutError, match="sleep_forever"):
            pool.run(sleep_forever, 0.5)
        # Pool should still work via spare
        assert pool.run(add, 5.0, a=1, b=2) == 3
    finally:
        pool.shutdown()


def test_multiple_rotations():
    pool = WarmPool(max_tasks=1, keep_spare=True)
    try:
        for i in range(5):
            assert pool.run(add, 5.0, a=i, b=1) == i + 1
    finally:
        pool.shutdown()


def test_spare_replenished():
    pool = WarmPool(max_tasks=1, keep_spare=True)
    try:
        assert pool.run(add, 5.0, a=1, b=1) == 2
        # After rotation, spare should be replenished
        assert pool._spare is not None
        assert pool._spare.process.is_alive()
    finally:
        pool.shutdown()


def test_rotation_faster_than_cold_start():
    # Time a cold-start pool creation
    start = time.perf_counter()
    cold = WarmPool(warming=warm_json_xml, keep_spare=False)
    cold_time = time.perf_counter() - start
    cold.shutdown()
    assert cold_time > 0

    # Create a pool with spare already warming
    pool = WarmPool(max_tasks=1, keep_spare=True, warming=warm_json_xml)
    # Run a task to exhaust active and trigger promotion
    assert pool.run(add, 5.0, a=1, b=1) == 2
    # Give spare time to be ready
    time.sleep(0.5)
    # Time the promotion
    start = time.perf_counter()
    assert pool.run(add, 5.0, a=2, b=2) == 4
    spare_time = time.perf_counter() - start
    pool.shutdown()
    assert spare_time > 0

    # Spare promotion should be faster than cold start
    assert spare_time < cold_time


def test_spare_hung_during_import_is_killed():
    """Spare that never sends 'ready' is killed on promotion and replaced
    with a cold-started worker.
    """
    pool = WarmPool(max_tasks=1, keep_spare=True, ready_timeout=0.2)
    try:
        # Replace the spare with a process that never sends "ready".
        pool._shutdown_worker(pool._spare)
        parent_connection, child_connection = Pipe()
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=hung_worker_process, args=(child_connection, 10)
        )
        process.start()
        hung_pid = process.pid
        pool._spare = WorkerHandle(
            process=process,
            connection=parent_connection,
            child_connection=child_connection,
            ready=False,
        )

        # Exhaust the active worker to trigger rotation → promotion.
        assert pool.run(add, 5.0, a=1, b=1) == 2

        # The hung spare should have been killed (not promoted).
        assert pool._active is not None
        assert pool._active.process.pid != hung_pid
        assert pool._active.ready

        # Verify the hung process was actually killed.
        assert (
            not psutil.pid_exists(hung_pid)
            or psutil.Process(hung_pid).status() == psutil.STATUS_ZOMBIE
        ), f"Hung process {hung_pid} was not killed"

        # Pool should still work after recovering.
        assert pool.run(add, 5.0, a=2, b=2) == 4
    finally:
        pool.shutdown()


def test_wait_for_ready_returns_false_on_timeout():
    """_wait_for_ready returns False when the worker never sends 'ready'."""
    pool = WarmPool(max_tasks=1, keep_spare=False, ready_timeout=0.2)
    try:
        parent_connection, child_connection = Pipe()
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=hung_worker_process, args=(child_connection, 10)
        )
        process.start()
        handle = WorkerHandle(
            process=process,
            connection=parent_connection,
            child_connection=child_connection,
            ready=False,
        )

        result = pool._wait_for_ready(handle)
        assert result is False
        assert handle.ready is False

        pool._kill_worker(handle)
    finally:
        pool.shutdown()
