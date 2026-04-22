## 2026-04-22 - Hardcoded JWT Secret Removed
**Vulnerability:** A hardcoded default JWT secret 'super-dev-secret-change-me-in-prod' was present in src/auth/grpc_server.py, which could be exploited if deployed.
**Learning:** Hardcoded fallback secrets must not be used even in fallback mechanisms.
**Prevention:** Rely strictly on validated secure application configuration (e.g., pydantic BaseSettings) to enforce that critical secrets are explicitly provided via the environment.
