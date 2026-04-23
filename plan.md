1. **Fix hardcoded secret in `src/auth/grpc_server.py`**:
   - The file `src/auth/grpc_server.py` contains a hardcoded fallback for the `JWT_SECRET_KEY` variable:
     `JWT_SECRET_KEY = os.getenv("JWT_SECRET", "super-dev-secret-change-me-in-prod")`
   - I will replace this with `JWT_SECRET_KEY = settings.JWT_SECRET` by importing `settings` from `src.shared.config`, which securely loads the secret from environment variables.
2. **Update Sentinel Journal**:
   - Add a critical learning entry to `.jules/sentinel.md` noting that relying on `os.getenv` with an insecure hardcoded fallback for cryptographic keys can lead to vulnerabilities in production environments. We should enforce centralized configuration via Pydantic settings.
3. **Run Unit Tests**:
   - Run `uv run pytest tests/unit/ -k "auth"` to ensure my changes do not break authentication functionality.
4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done**:
   - Run `pre_commit_instructions` and follow the steps.
5. **Submit PR**:
   - Commit the changes and open a PR with the format `🛡️ Sentinel: [CRITICAL] Fix hardcoded JWT secret fallback`.
