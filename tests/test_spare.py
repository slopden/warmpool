import time

import pytest

from warmpool import PoolWithTimeout

from ._helpers import add, get_pid, sleep_forever


def test_spare_created_on_init(spare_pool):
    assert spare_pool._spare is not None
    assert spare_pool._spare.process.is_alive()


def test_auto_rotation_on_exhaustion():
    p = PoolWithTimeout(max_tasks=2, keep_spare=True)
    try:
        p.run(add, 5.0, a=1, b=1)
        pid2 = p.run(get_pid, 5.0)
        # After 2 tasks the active worker should rotate
        pid3 = p.run(get_pid, 5.0)
        assert pid3 != pid2
    finally:
        p.shutdown()


def test_rotation_after_timeout():
    p = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        with pytest.raises(TimeoutError):
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
        p.run(add, 5.0, a=1, b=1)
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

    # Create a pool with spare already warming
    p = PoolWithTimeout(max_tasks=1, keep_spare=True, warm_modules=["json", "xml"])
    # Run a task to exhaust active and trigger promotion
    p.run(add, 5.0, a=1, b=1)
    # Give spare time to be ready
    time.sleep(0.5)
    # Time the promotion
    t0 = time.perf_counter()
    p.run(add, 5.0, a=2, b=2)
    spare_time = time.perf_counter() - t0
    p.shutdown()

    # Spare promotion should be faster than cold start
    assert spare_time < cold_time
