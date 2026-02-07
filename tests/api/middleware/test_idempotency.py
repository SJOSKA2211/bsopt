import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.middleware.idempotency import IdempotencyMiddleware

app = FastAPI()


@app.post("/test")
async def mock_endpoint(request: Request):
    return {"message": "Success"}


mock_redis_client = AsyncMock()

app.add_middleware(
    IdempotencyMiddleware, redis_client=mock_redis_client, expiry=10, lock_timeout=1
)

client = TestClient(app)


def test_idempotency_skipped_for_get():
    mock_redis_client.get.reset_mock()
    response = client.get("/")
    mock_redis_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_idempotency_cache_hit():
    cached_content = json.dumps({"message": "Cached"})
    cached_response = {"content": cached_content, "status_code": 200, "headers": {}}
    mock_redis_client.get.return_value = json.dumps(cached_response).encode()

    response = client.post("/test")

    assert response.status_code == 200
    assert "HIT" in response.headers["X-Idempotency-Cache"]
    assert response.json() == {"message": "Cached"}


@pytest.mark.asyncio
async def test_idempotency_cache_miss_and_set():
    mock_redis_client.get.reset_mock()
    mock_redis_client.set.reset_mock()
    mock_redis_client.get.return_value = None
    mock_redis_client.setnx.return_value = True

    response = client.post("/test")

    assert response.status_code == 200
    assert response.json() == {"message": "Success"}
    assert mock_redis_client.set.called


@pytest.mark.asyncio
async def test_idempotency_lock_conflict():
    mock_redis_client.get.return_value = None
    mock_redis_client.setnx.return_value = False

    response = client.post("/test")

    assert response.status_code == 409
    assert "Request already in progress" in response.text
