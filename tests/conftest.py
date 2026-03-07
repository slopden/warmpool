import pytest

from warmpool import PoolWithTimeout


@pytest.fixture
def pool():
    p = PoolWithTimeout(max_tasks=50, keep_spare=False)
    yield p
    p.shutdown()


@pytest.fixture
def spare_pool():
    p = PoolWithTimeout(max_tasks=50)
    yield p
    p.shutdown()
