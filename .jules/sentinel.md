## 2024-04-24 - Unauthenticated Mock Bypass
**Vulnerability:** The API allowed unauthenticated mock bypass using the `X-User-ID` header regardless of the environment.
**Learning:** Development-only mock bypasses must be explicitly gated behind environment checks.
**Prevention:** Always use `settings.ENVIRONMENT != "production"` when adding development bypass logic.
