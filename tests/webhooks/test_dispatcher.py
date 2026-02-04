import pytest
import hmac
import hashlib
import time

from src.webhooks.dispatcher import _generate_signature, _verify_signature

@pytest.mark.asyncio
async def test_generate_signature_consistency():
    secret = "webhook_secret"
    payload = '{"id": "evt_test", "type": "event.test"}'
    timestamp = int(time.time())
    
    sig1 = await _generate_signature(secret, payload, timestamp)
    sig2 = await _generate_signature(secret, payload, timestamp)
    
    assert sig1 == sig2
    assert sig1.startswith("t=")
    assert "sha256=" in sig1

@pytest.mark.asyncio
async def test_verify_signature_valid():
    secret = "webhook_secret"
    payload = '{"id": "evt_test", "type": "event.test"}'
    timestamp = int(time.time())
    
    signature_header = await _generate_signature(secret, payload, timestamp)
    
    # Extract timestamp and signature from the header
    parts = signature_header.split(',')
    t_part = next(part for part in parts if part.startswith("t="))
    sha256_part = next(part for part in parts if part.startswith("sha256="))
    
    extracted_timestamp = int(t_part.split('=')[1])
    extracted_signature = sha256_part.split('=')[1]
    
    assert await _verify_signature(secret, payload, extracted_timestamp, extracted_signature) is True

@pytest.mark.asyncio
async def test_verify_signature_invalid_payload():
    secret = "webhook_secret"
    payload = '{"id": "evt_test", "type": "event.test"}'
    timestamp = int(time.time())
    
    signature_header = await _generate_signature(secret, payload, timestamp)
    
    parts = signature_header.split(',')
    t_part = next(part for part in parts if part.startswith("t="))
    sha256_part = next(part for part in parts if part.startswith("sha256="))
    
    extracted_timestamp = int(t_part.split('=')[1])
    extracted_signature = sha256_part.split('=')[1]
    
    invalid_payload = '{"id": "evt_test", "type": "event.wrong"}'
    
    assert await _verify_signature(secret, invalid_payload, extracted_timestamp, extracted_signature) is False

@pytest.mark.asyncio
async def test_verify_signature_invalid_secret():
    secret = "webhook_secret"
    payload = '{"id": "evt_test", "type": "event.test"}'
    timestamp = int(time.time())
    
    signature_header = await _generate_signature(secret, payload, timestamp)
    
    parts = signature_header.split(',')
    t_part = next(part for part in parts if part.startswith("t="))
    sha256_part = next(part for part in parts if part.startswith("sha256="))
    
    extracted_timestamp = int(t_part.split('=')[1])
    extracted_signature = sha256_part.split('=')[1]
    
    invalid_secret = "wrong_secret"
    
    assert await _verify_signature(invalid_secret, payload, extracted_timestamp, extracted_signature) is False

@pytest.mark.asyncio
async def test_verify_signature_expired_timestamp():
    secret = "webhook_secret"
    payload = '{"id": "evt_test", "type": "event.test"}'
    timestamp = int(time.time()) - 600 # 10 minutes ago
    
    signature_header = await _generate_signature(secret, payload, timestamp)
    
    parts = signature_header.split(',')
    t_part = next(part for part in parts if part.startswith("t="))
    sha256_part = next(part for part in parts if part.startswith("sha256="))
    
    extracted_timestamp = int(t_part.split('=')[1])
    extracted_signature = sha256_part.split('=')[1]
    
    # Assuming a tolerance of 5 minutes (300 seconds)
    assert await _verify_signature(secret, payload, extracted_timestamp, extracted_signature, tolerance=300) is False

@pytest.mark.asyncio
async def test_verify_signature_future_timestamp():
    secret = "webhook_secret"
    payload = '{"id": "evt_test", "type": "event.test"}'
    timestamp = int(time.time()) + 600 # 10 minutes in future
    
    signature_header = await _generate_signature(secret, payload, timestamp)
    
    parts = signature_header.split(',')
    t_part = next(part for part in parts if part.startswith("t="))
    sha256_part = next(part for part in parts if part.startswith("sha256="))
    
    extracted_timestamp = int(t_part.split('=')[1])
    extracted_signature = sha256_part.split('=')[1]
    
    # Assuming a tolerance of 5 minutes (300 seconds)
    assert await _verify_signature(secret, payload, extracted_timestamp, extracted_signature, tolerance=300) is False