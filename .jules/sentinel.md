## 2024-04-24 - Fix Auth Bypass in Production

**Vulnerability:** A development testing bypass using the `X-User-ID` header to bypass authentication checks was left unguarded in `get_current_user` in `src/api/dependencies.py`. Any user could use this header in production to impersonate another user and bypass the gRPC auth service.
**Learning:** Development testing bypasses should strictly be constrained to non-production environments to avoid leaving critical holes in the final builds. They were meant for local dev testing but weren't removed or guarded properly.
**Prevention:** Always verify that such mock endpoints or mock headers are gated behind an `if settings.ENVIRONMENT != "production":` check. Never leave them open to the internet.
