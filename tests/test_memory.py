import psutil

from warmpool import PoolStatus, PoolWithTimeout

from ._helpers import add, allocate_memory, get_pid


def test_memory_rotation_absolute():
    """Worker PID changes after RSS exceeds max_memory."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=1)
    try:
        pid1 = pool.run(get_pid, 5.0)
        pool.run(allocate_memory, 5.0, megabytes=1)
        pid2 = pool.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        pool.shutdown()


def test_memory_rotation_percent():
    """Worker PID changes after RSS exceeds max_memory_percent."""
    # Use a tiny fraction so any worker exceeds it.
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory_percent=0.0)
    try:
        pid1 = pool.run(get_pid, 5.0)
        pid2 = pool.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        pool.shutdown()


def test_memory_no_rotation_under_limit():
    """Same PID when RSS stays under the memory limit."""
    total = psutil.virtual_memory().total
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=total)
    try:
        pid1 = pool.run(get_pid, 5.0)
        pool.run(add, 5.0, a=1, b=2)
        pid2 = pool.run(get_pid, 5.0)
        assert pid2 == pid1
    finally:
        pool.shutdown()


def test_memory_rotation_without_spare():
    """keep_spare=False + memory exceeded -> PoolStatus.EXHAUSTED."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=False, max_memory=1)
    try:
        pool.run(allocate_memory, 5.0, megabytes=1)
        assert pool.status is PoolStatus.EXHAUSTED
    finally:
        pool.shutdown()


def test_memory_both_limits_absolute_triggers():
    """Absolute limit triggers even when percent is permissive."""
    pool = PoolWithTimeout(
        max_tasks=50, keep_spare=True, max_memory=1, max_memory_percent=1.0
    )
    try:
        pid1 = pool.run(get_pid, 5.0)
        pool.run(allocate_memory, 5.0, megabytes=1)
        pid2 = pool.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        pool.shutdown()


def test_memory_default_no_check():
    """Default (None/None) never rotates even with high RSS."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        pid1 = pool.run(get_pid, 5.0)
        pool.run(allocate_memory, 5.0, megabytes=10)
        pid2 = pool.run(get_pid, 5.0)
        assert pid2 == pid1
    finally:
        pool.shutdown()


def test_last_memory_rss_populated():
    """last_memory_rss is int > 0 after task with memory checking on."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=1024 * 1024 * 500)
    try:
        pool.run(add, 5.0, a=1, b=2)
        assert isinstance(pool.last_memory_rss, int)
        assert pool.last_memory_rss > 0
    finally:
        pool.shutdown()


def test_last_memory_rss_none_when_disabled():
    """last_memory_rss is None when memory checking is off."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        pool.run(add, 5.0, a=1, b=2)
        assert pool.last_memory_rss is None
    finally:
        pool.shutdown()


def test_memory_rotation_preserves_elapsed_ms():
    """last_elapsed_ms still set when memory rotation triggers."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=1)
    try:
        pool.run(allocate_memory, 5.0, megabytes=1)
        assert isinstance(pool.last_elapsed_ms, int)
        assert pool.last_elapsed_ms >= 0
    finally:
        pool.shutdown()


def test_memory_rotation_zero_limit():
    """max_memory=0 rotates on every task (any RSS exceeds 0)."""
    pool = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=0)
    try:
        pid1 = pool.run(get_pid, 5.0)
        pid2 = pool.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        pool.shutdown()
