#!/usr/bin/env python
"""Demonstrate warmpool's hard-kill timeout.

Shows three scenarios:
  1. A worker that logs from inside the subprocess, then hangs in
     ``time.sleep`` — killed after 1 second.
  2. A worker stuck in LAPACK C code (``scipy.linalg.eigh`` on a huge
     matrix) where Python signal handlers never fire — only SIGKILL
     works.  Killed after 0.5 seconds.
  3. Normal recovery: the pool rotates to the spare and runs a
     subsequent task successfully after each kill.

Run with:
    uv run python example.py
"""

import logging
import time

from warmpool import PoolWithTimeout

# ── Set up root logger so we see both parent and forwarded child logs ──
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("example")


# ── Functions that run inside the subprocess ──
# (must be module-level so `spawn` can pickle them)


def hang_after_log():
    """Log a warning from inside the worker, then sleep forever."""
    import logging
    import time

    start = time.perf_counter()
    import numpy as np  # already warm — no import cost

    logging.getLogger("worker.task").warning(
        f"I'm inside the process! numpy={np.__version__} and it took {time.perf_counter() - start:0.4f}s to import because we warmed up!"
    )
    time.sleep(86400)  # 24 hours — will be killed long before this


def eigh_huge(n=5000):
    """Eigendecompose a large symmetric matrix — stuck in LAPACK C code."""
    import numpy as np
    from scipy import linalg

    a = np.random.rand(n, n)
    a = a + a.T  # symmetric so eigh is valid
    return linalg.eigh(a)


def add(a=0, b=0):
    return a + b


def warm_imports():
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import scipy.linalg  # noqa: F401


def main():
    log.info("creating pool with keep_spare=True, warming numpy+scipy")
    pool = PoolWithTimeout(
        warming=warm_imports,
        max_tasks=50,
        keep_spare=True,
    )

    # ── Scenario 1: kill a worker hanging in Python sleep ──
    log.info("dispatching hang_after_log() with 1s timeout …")
    start = time.perf_counter()
    try:
        pool.run(hang_after_log, timeout=1.0)
    except TimeoutError:
        elapsed = time.perf_counter() - start
        log.info(f"worker killed after {elapsed:.2f}s (TimeoutError)")

    # Pool recovered — prove it
    result = pool.run(add, timeout=5.0, a=2, b=3)
    log.info(f"recovered: add(2, 3) = {result}")

    # ── Scenario 2: kill a worker stuck in C code (LAPACK eigh) ──
    log.info("dispatching eigh_huge(5000) — stuck in LAPACK C, 0.5s timeout …")
    start = time.perf_counter()
    try:
        pool.run(eigh_huge, timeout=0.5, n=5000)
    except TimeoutError:
        elapsed = time.perf_counter() - start
        log.info(f"LAPACK worker killed after {elapsed:.2f}s (TimeoutError)")

    # Pool recovered again
    result = pool.run(add, timeout=5.0, a=7, b=8)
    log.info(f"recovered: add(7, 8) = {result}")

    pool.shutdown()
    log.info("done")


if __name__ == "__main__":
    main()
