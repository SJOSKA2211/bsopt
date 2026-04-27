## 2024-11-23 - Hardcoded Fallback JWT Secret in Auth Service
**Vulnerability:** A hardcoded, insecure development fallback secret ("super-dev-secret-change-me-in-prod") was used for `JWT_SECRET_KEY` in `src/auth/grpc_server.py`.
**Learning:** Even when a centralized `settings` module enforces strict configuration rules (like `JWT_SECRET: str = Field(...)`), local file imports might bypass it by using raw `os.getenv` with fallback defaults, silently creating vulnerabilities if the environment variable is missing.
**Prevention:** Always use the centralized `settings` module (e.g., `from src.shared.config import settings`) that handles validation for cryptographic secrets. Never supply insecure fallback strings in application code.
