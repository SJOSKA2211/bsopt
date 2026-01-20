import pytest
from unittest.mock import AsyncMock, MagicMock
from src.api.middleware.idempotency import _generate_fingerprint

@pytest.mark.asyncio
async def test_fingerprint_consistency():
    # Mock FastAPI request
    request = MagicMock()
    request.method = "POST"
    request.url.path = "/api/v1/trade"
    request.headers = {"X-Idempotency-Key": "unique-key", "Content-Type": "application/json"}
    
    body = b'{"symbol": "AAPL", "quantity": 10}'
    request.body = AsyncMock(return_value=body)
    
    fp1 = await _generate_fingerprint(request)
    
    # Second request with identical data
    request2 = MagicMock()
    request2.method = "POST"
    request2.url.path = "/api/v1/trade"
    request2.headers = {"X-Idempotency-Key": "unique-key", "Content-Type": "application/json"}
    request2.body = AsyncMock(return_value=body)
    
    fp2 = await _generate_fingerprint(request2)
    
    assert fp1 == fp2
    assert len(fp1) == 64 # SHA-256 hex length

@pytest.mark.asyncio
async def test_fingerprint_differs_on_body():
    request1 = MagicMock()
    request1.method = "POST"
    request1.url.path = "/api/v1/trade"
    request1.headers = {"X-Idempotency-Key": "key"}
    request1.body = AsyncMock(return_value=b'{"q": 10}')
    
    request2 = MagicMock()
    request2.method = "POST"
    request2.url.path = "/api/v1/trade"
    request2.headers = {"X-Idempotency-Key": "key"}
    request2.body = AsyncMock(return_value=b'{"q": 20}')
    
    fp1 = await _generate_fingerprint(request1)
    fp2 = await _generate_fingerprint(request2)
    
    assert fp1 != fp2

@pytest.mark.asyncio
async def test_fingerprint_differs_on_path():
    request1 = MagicMock()
    request1.method = "POST"
    request1.url.path = "/path1"
    request1.headers = {"X-Idempotency-Key": "key"}
    request1.body = AsyncMock(return_value=b'{}')
    
    request2 = MagicMock()
    request2.method = "POST"
    request2.url.path = "/path2"
    request2.headers = {"X-Idempotency-Key": "key"}
    request2.body = AsyncMock(return_value=b'{}')
    
    fp1 = await _generate_fingerprint(request1)
    fp2 = await _generate_fingerprint(request2)
    
    assert fp1 != fp2

@pytest.mark.asyncio
async def test_fingerprint_header_insensitivity():
    request1 = MagicMock()
    request1.headers = {"X-IDEMPOTENCY-KEY": "key"}
    request1.method = "POST"
    request1.url.path = "/test"
    request1.body = AsyncMock(return_value=b'{}')
    
    request2 = MagicMock()
    request2.headers = {"x-idempotency-key": "key"}
    request2.method = "POST"
    request2.url.path = "/test"
    request2.body = AsyncMock(return_value=b'{}')
    
    fp1 = await _generate_fingerprint(request1)
    fp2 = await _generate_fingerprint(request2)
    
    assert fp1 == fp2
