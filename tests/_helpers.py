"""Module-level picklable functions for subprocess tests.

These must be module-level (not closures or lambdas) so that the
``spawn`` multiprocessing context can pickle them.
"""

import logging
import os
import time


def add(a=0, b=0):
    return a + b


def identity(x=None):
    return x


def get_pid():
    return os.getpid()


def sleep_forever():
    time.sleep(3600)
    raise ValueError("Should have been killed long ago!")


def sleep_seconds(seconds=1.0):
    time.sleep(seconds)


def raise_value_error(msg="test error"):
    raise ValueError(msg)


def log_message(msg="hello", level="INFO"):
    lvl = getattr(logging, level, logging.INFO)
    logging.getLogger("test.helper").log(lvl, msg)


def log_with_exception():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("test.helper").exception("caught error")


def sleep_short(seconds=0.01):
    time.sleep(seconds)


def check_module_imported(module_name="json"):
    import sys

    return module_name in sys.modules


def force_exit(code=1):
    os._exit(code)


def raise_unpicklable():
    """Raise an exception whose class can't be pickled.

    Simulates C-API wrappers that produce unpicklable exception types.
    """

    class _Unpicklable(Exception):
        """Locally defined so pickle can't resolve the class."""

        pass

    raise _Unpicklable("this can't be pickled")


_memory_holder = None


def allocate_memory(mb=100):
    """Allocate ~mb megabytes of RSS that persists in the worker process."""
    global _memory_holder
    _memory_holder = bytearray(mb * 1024 * 1024)
    return len(_memory_holder)


def scipy_eigh_huge(n=5000):
    """Compute eigenvalues of a large symmetric matrix.

    Runs entirely inside LAPACK C code (holds the GIL), so Python
    signal handlers never get a chance to fire — only SIGKILL works.
    """
    import numpy as np
    from scipy import linalg

    a = np.random.rand(n, n)
    a = a + a.T
    return linalg.eigh(a)
