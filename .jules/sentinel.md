## 2024-04-19 - Removed Hardcoded Secret Key from Auth Service

**Vulnerability:** A hardcoded secret key `JWT_SECRET_KEY = os.getenv("JWT_SECRET", "super-dev-secret-change-me-in-prod")` was found in `src/auth/grpc_server.py`.
**Learning:** Development defaults for secrets should never be checked in as string literals in production code, even as fallbacks, to prevent inadvertent exposure or reliance on default insecure keys.
**Prevention:** Remove insecure fallbacks entirely or enforce strict validation via configurations (e.g., Pydantic settings) that fail on missing secrets rather than defaulting to insecure values.
