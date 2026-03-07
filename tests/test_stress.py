import pytest

from warmpool import PoolWithTimeout

from ._helpers import add, sleep_forever


def test_stress_mixed_calls():
    """1000 calls with max_tasks=50, mixing success and timeout."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        successes = 0
        timeouts = 0
        for i in range(1000):
            if i % 50 == 49:
                # Every 50th call: trigger a timeout
                with pytest.raises(TimeoutError):
                    p.run(sleep_forever, 0.2)
                timeouts += 1
            else:
                result = p.run(add, 5.0, a=i, b=1)
                assert result == i + 1
                successes += 1

        assert successes == 980
        assert timeouts == 20
        # Pool should still be functional
        assert p.run(add, 5.0, a=1, b=2) == 3
    finally:
        p.shutdown()


def test_stress_rapid_fire():
    """1000 fast calls to verify no resource leaks or pipe desync."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        for i in range(1000):
            assert p.run(add, 5.0, a=i, b=i) == i * 2
    finally:
        p.shutdown()
