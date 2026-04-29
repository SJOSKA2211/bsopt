## 2024-05-24 - Hardcoded JWT Secret in Auth Server
**Vulnerability:** A hardcoded fallback JWT secret ("super-dev-secret-change-me-in-prod") was present in `src/auth/grpc_server.py`.
**Learning:** Hardcoded fallback values for secrets are dangerous because they can easily slip into production if environment variables are misconfigured. It should rely on secure configuration management that strictly enforces the presence of secrets.
**Prevention:** Ensure all sensitive configuration values are fetched securely and without fallback values. Use robust configuration classes (like Pydantic Settings) that fail early if a required secret is missing.
