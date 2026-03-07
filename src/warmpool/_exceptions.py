"""Exception types for the warmpool package."""


class ProcessPoolExhausted(Exception):
    """The worker process is dead or has hit its task limit.

    When ``keep_spare=False``, the caller must create a new
    :class:`~warmpool.PoolWithTimeout` instance to continue.

    Parameters
    ----------
    message
        Human-readable description of why the pool is exhausted.
    exit_code
        The worker's exit code, if it died.  ``None`` when the pool
        was shut down explicitly or the code is unavailable.
    """

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code
