import pytest

from warmpool import WarmPool


@pytest.fixture
def pool():
    pool = WarmPool(max_tasks=50, keep_spare=False)
    yield pool
    pool.shutdown()


@pytest.fixture
def spare_pool():
    pool = WarmPool(max_tasks=50)
    yield pool
    pool.shutdown()
