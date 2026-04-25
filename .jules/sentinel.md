## 2024-04-25 - Remove hardcoded secret fallback
**Vulnerability:** Found a hardcoded fallback string for `JWT_SECRET_KEY` in `src/auth/grpc_server.py`.
**Learning:** Hardcoded fallbacks in `os.getenv` can lead to insecure production deployments if the environment variable is missing. It's safer to rely on Pydantic `BaseSettings` which enforces required variables.
**Prevention:** Always use centralized configuration objects (like `settings`) that raise `ValidationError` when critical secrets are missing, rather than providing insecure string fallbacks.
