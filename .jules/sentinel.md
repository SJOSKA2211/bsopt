## 2024-05-20 - Remove hardcoded JWT Secret Fallback
**Vulnerability:** Found a hardcoded fallback string for the JWT secret key `os.getenv("JWT_SECRET", "super-dev-secret-change-me-in-prod")` in `src/auth/grpc_server.py`.
**Learning:** Hardcoding fallback secrets bypasses environment variable requirements and could lead to critical unauthorized access if deployed without proper environment configuration. The application needs to fail securely on startup if missing critical secrets.
**Prevention:** Use validated configuration files (like Pydantic's `BaseSettings` used in `src.shared.config`) to enforce that critical secrets like `JWT_SECRET` are always provided.
