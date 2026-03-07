import psutil

from warmpool import PoolStatus, PoolWithTimeout

from ._helpers import add, allocate_memory, get_pid


def test_memory_rotation_absolute():
    """Worker PID changes after RSS exceeds max_memory."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=1)
    try:
        pid1 = p.run(get_pid, 5.0)
        p.run(allocate_memory, 5.0, mb=1)
        pid2 = p.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        p.shutdown()


def test_memory_rotation_percent():
    """Worker PID changes after RSS exceeds max_memory_percent."""
    # Use a tiny fraction so any worker exceeds it.
    p = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory_percent=0.0)
    try:
        pid1 = p.run(get_pid, 5.0)
        pid2 = p.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        p.shutdown()


def test_memory_no_rotation_under_limit():
    """Same PID when RSS stays under the memory limit."""
    total = psutil.virtual_memory().total
    p = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=total)
    try:
        pid1 = p.run(get_pid, 5.0)
        p.run(add, 5.0, a=1, b=2)
        pid2 = p.run(get_pid, 5.0)
        assert pid2 == pid1
    finally:
        p.shutdown()


def test_memory_rotation_without_spare():
    """keep_spare=False + memory exceeded -> PoolStatus.EXHAUSTED."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=False, max_memory=1)
    try:
        p.run(allocate_memory, 5.0, mb=1)
        assert p.status is PoolStatus.EXHAUSTED
    finally:
        p.shutdown()


def test_memory_both_limits_absolute_triggers():
    """Absolute limit triggers even when percent is permissive."""
    p = PoolWithTimeout(
        max_tasks=50, keep_spare=True, max_memory=1, max_memory_percent=1.0
    )
    try:
        pid1 = p.run(get_pid, 5.0)
        p.run(allocate_memory, 5.0, mb=1)
        pid2 = p.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        p.shutdown()


def test_memory_default_no_check():
    """Default (None/None) never rotates even with high RSS."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        pid1 = p.run(get_pid, 5.0)
        p.run(allocate_memory, 5.0, mb=10)
        pid2 = p.run(get_pid, 5.0)
        assert pid2 == pid1
    finally:
        p.shutdown()


def test_last_memory_rss_populated():
    """last_memory_rss is int > 0 after task with memory checking on."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=1024 * 1024 * 500)
    try:
        p.run(add, 5.0, a=1, b=2)
        assert isinstance(p.last_memory_rss, int)
        assert p.last_memory_rss > 0
    finally:
        p.shutdown()


def test_last_memory_rss_none_when_disabled():
    """last_memory_rss is None when memory checking is off."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True)
    try:
        p.run(add, 5.0, a=1, b=2)
        assert p.last_memory_rss is None
    finally:
        p.shutdown()


def test_memory_rotation_preserves_elapsed_ms():
    """last_elapsed_ms still set when memory rotation triggers."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=1)
    try:
        p.run(allocate_memory, 5.0, mb=1)
        assert isinstance(p.last_elapsed_ms, int)
        assert p.last_elapsed_ms >= 0
    finally:
        p.shutdown()


def test_memory_rotation_zero_limit():
    """max_memory=0 rotates on every task (any RSS exceeds 0)."""
    p = PoolWithTimeout(max_tasks=50, keep_spare=True, max_memory=0)
    try:
        pid1 = p.run(get_pid, 5.0)
        pid2 = p.run(get_pid, 5.0)
        assert pid2 != pid1
    finally:
        p.shutdown()
