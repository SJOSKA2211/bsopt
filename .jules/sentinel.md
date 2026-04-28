## 2026-04-28 - Development Auth Bypass in Production
**Vulnerability:** A mock development authentication bypass using the X-User-ID header was left active in production environments.
**Learning:** Development testing aids can easily slip into production if not strictly gated by environment checks.
**Prevention:** Always gate mock logic, test headers, and bypasses with strict environment checks (e.g., settings.ENVIRONMENT != 'production').
