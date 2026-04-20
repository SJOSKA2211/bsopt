## 2025-03-09 - Fixed Hardcoded JWT Secret Fallback

**Vulnerability:** A hardcoded default value ("super-dev-secret-change-me-in-prod") was present for `JWT_SECRET_KEY` in `src/auth/grpc_server.py`.
**Learning:** Hardcoded fallbacks in authentication code can easily be leaked into production if environment variables are not correctly set, posing a significant risk of token forgery and unauthorized access.
**Prevention:** Rely strictly on validated configuration objects like Pydantic's `Settings` which will fail to start if the required secret environment variables are not supplied. Do not include development secrets as fallback strings in the source code.
