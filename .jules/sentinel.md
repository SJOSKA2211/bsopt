## 2025-05-15 - Remove Hardcoded JWT Secret
**Vulnerability:** A hardcoded fallback secret 'super-dev-secret-change-me-in-prod' was present in src/auth/grpc_server.py.
**Learning:** Insecure default credentials can bypass environment variable requirements and leak into production. Always enforce strict loading from a central settings configuration instead of providing inline fallbacks.
**Prevention:** Rely strictly on the Pydantic Settings model which enforces required secrets at startup.
