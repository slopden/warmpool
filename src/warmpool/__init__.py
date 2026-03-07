"""warmpool — single-worker subprocess pool with hard-kill timeouts.

Usage
-----
>>> from warmpool import PoolWithTimeout, PoolStatus
>>> pool = PoolWithTimeout(max_tasks=100, keep_spare=True)
>>> result = pool.run(my_func, timeout=10.0, x=42)
>>> pool.status is PoolStatus.READY
True
>>> pool.shutdown()

Memory-based rotation:

>>> pool = PoolWithTimeout(max_memory=500 * 1024 * 1024)
"""

from ._exceptions import ProcessPoolExhausted
from .pool import PoolStatus, PoolWithTimeout

__all__ = ["PoolStatus", "PoolWithTimeout", "ProcessPoolExhausted"]
