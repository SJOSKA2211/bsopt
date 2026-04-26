## 2024-05-18 - Remove Hardcoded JWT Secret Default
**Vulnerability:** A hardcoded default secret ("super-dev-secret-change-me-in-prod") was used for `JWT_SECRET_KEY` in `src/auth/grpc_server.py` as a fallback for missing environment variables.
**Learning:** Relying on hardcoded fallbacks for cryptographic secrets is dangerous because if the environment variable is accidentally omitted in production, the application will silently fall back to an insecure, known secret, compromising all JWTs.
**Prevention:** Always enforce the presence of required cryptographic secrets at startup (e.g., using a configuration manager like Pydantic BaseSettings without defaults) and remove hardcoded fallback values from application code.
