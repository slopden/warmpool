"""warmpool — single-worker subprocess pool with hard-kill timeouts.

Usage
-----
>>> from warmpool import PoolWithTimeout, PoolStatus
>>> pool = PoolWithTimeout(max_tasks=100, keep_spare=True)
>>> result = pool.run(my_func, timeout=10.0, x=42)
>>> pool.status is PoolStatus.READY
True
>>> pool.shutdown()
"""

from ._exceptions import ProcessPoolExhausted
from .pool import PoolStatus, PoolWithTimeout

__all__ = ["PoolStatus", "PoolWithTimeout", "ProcessPoolExhausted"]
