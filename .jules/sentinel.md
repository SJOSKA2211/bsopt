## 2025-05-27 - Rate Limiting Pattern & Testing Mocks

**Vulnerability:** Missing rate limiting on sensitive endpoints (Login/Register) allowing brute force/DoS.
**Learning:**
1.  The `RateLimiter` dependency was originally a function, making it hard to customize per-endpoint. Converted to a class for flexibility while maintaining backward compatibility.
2.  Tests rely heavily on `conftest.py` mocks. `numba` and `redis` caused import/execution errors in tests because they were either not mocked or mocked incorrectly for certain contexts (e.g., missing `get_redis_client` override).
**Prevention:**
1.  Use `RateLimiter(requests=N, window=M)` dependency for sensitive endpoints.
2.  Ensure `sys.modules["numba"] = MagicMock()` is in `conftest.py` to avoid build dependency issues in CI/Test environments.
