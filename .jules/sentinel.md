## 2024-04-23 - Hardcoded JWT Secret Fallback
**Vulnerability:** Found a hardcoded fallback for JWT_SECRET_KEY in src/auth/grpc_server.py (`os.getenv("JWT_SECRET", "super-dev-secret-change-me-in-prod")`).
**Learning:** Relying on `os.getenv` with an insecure hardcoded fallback for cryptographic keys can lead to vulnerabilities in production environments if the environment variable is accidentally omitted.
**Prevention:** We should enforce centralized configuration via Pydantic settings (`src.shared.config.settings`), which validates that critical secrets are explicitly provided.
