import pytest

from warmpool import PoolWithTimeout


@pytest.fixture
def pool():
    pool = PoolWithTimeout(max_tasks=50, keep_spare=False)
    yield pool
    pool.shutdown()


@pytest.fixture
def spare_pool():
    pool = PoolWithTimeout(max_tasks=50)
    yield pool
    pool.shutdown()
