import pytest

from src.shared import observability


def test_setup_logging_idempotent():
    # It takes no arguments!
    observability.setup_logging()
    assert True


def test_tune_gc():
    # Should run without error
    observability.tune_gc(mode="high_throughput")
    observability.tune_gc(mode="analytical")
    assert True


def test_tune_worker_resources():
    # Should run without error
    observability.tune_worker_resources()
    assert True


@pytest.mark.asyncio
async def test_get_obs_client():
    client = observability.get_obs_client()
    assert client is not None
    await client.aclose()
