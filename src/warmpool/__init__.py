"""warmpool — single-worker subprocess pool with hard-kill timeouts.

Usage
-----
>>> from warmpool import WarmPool, PoolStatus
>>> pool = WarmPool(max_tasks=100, keep_spare=True)
>>> result = pool.run(my_func, timeout=10.0, x=42)
>>> pool.status is PoolStatus.READY
True
>>> pool.shutdown()

Memory-based rotation:

>>> pool = WarmPool(max_memory=500 * 1024 * 1024)
"""

from ._exceptions import ProcessPoolExhausted
from .pool import PoolStatus, WarmPool

__all__ = ["PoolStatus", "WarmPool", "ProcessPoolExhausted"]
