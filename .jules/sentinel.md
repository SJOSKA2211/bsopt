## 2025-02-14 - Unused Security Middleware and Missing Rate Limits
**Vulnerability:** The `/login` endpoint had no rate limiting, exposing it to brute force attacks. Additionally, `IPBlockMiddleware` was defined in the codebase but not added to the application middleware stack.
**Learning:** Existence of security code (like `IPBlockMiddleware`) does not guarantee it is active. Always verify that middleware is actually mounted in the application entry point.
**Prevention:** Audit `main.py` or application factories to ensure all security middleware is correctly registered. Use integration tests to verify security controls are active.

## 2025-02-14 - Rate Limiting Vulnerabilities
**Vulnerability:** The `/login` endpoint lacked rate limiting, and the initial design considered using `X-Forwarded-For` which is vulnerable to spoofing.
**Learning:** Trusting `X-Forwarded-For` blindly allows attackers to bypass IP-based rate limits. Also, non-atomic Redis operations can lead to race conditions where keys never expire.
**Prevention:** Use `request.client.host` and rely on infrastructure to handle proxy headers. Use Redis Lua scripts for atomic `INCR` + `EXPIRE` operations.
