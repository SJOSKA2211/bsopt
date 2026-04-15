## 2026-02-23 - [Missing Security Middleware in API]
**Vulnerability:** The API had defined but unused security middlewares (`SecurityHeadersMiddleware`, `InputSanitizationMiddleware`). While `JWTAuthenticationMiddleware` was used, headers like HSTS, CSP, and X-Frame-Options were missing, leaving the application vulnerable to clickjacking and MIME sniffing.
**Learning:** Middleware definitions are not enough; always verify their integration in the main application entry point (`main.py`). The separation of definition (`src/api/middleware/security.py`) and usage (`src/api/main.py`) led to this gap.
**Prevention:** Use integration tests that explicitly check for security headers on public endpoints (like `/health`) to catch this regression early.

## 2026-02-25 - [Broken Access Control in User Listing]
**Vulnerability:** The `list_users` endpoint was accessible to any authenticated user, allowing Privilege Escalation (Regular User -> Admin Read Access) and IDOR.
**Learning:** `JWTAuthenticationMiddleware` only ensures *authentication* (who you are), not *authorization* (what you can do). Endpoints returning sensitive data must explicitly check roles/tiers.
**Prevention:** Audit all endpoints returning collections or sensitive resources for `require_tier` or role-based dependencies. Enforce "default deny" authorization where possible or use linter rules to flag missing `Depends` on sensitive routes.
## 2026-03-01 - [Missing Security Middleware in API - CSRF and IP Blocking]
**Vulnerability:** The API had defined but unused security middlewares (`IPBlockMiddleware`, `CSRFMiddleware`). These were omitted from the main application startup file, leaving the application vulnerable to IP-based attacks (e.g., brute force, DDoS) and CSRF attacks for state-changing endpoints.
**Learning:** Adding new security middleware to the repository is not sufficient; they must be explicitly registered to the application's middleware stack in `main.py` in the correct order.
**Prevention:** Always verify integration of new security middleware definitions in the main application entry point (`main.py`). Consider enforcing this through an automated linting or testing rule that ensures all exported security middleware classes are applied.
## 2024-11-06 - [DoS via Double Argon2 Penalty on Invalid Logins]
**Vulnerability:** The `authenticate_user` function dynamically generated a new Argon2 hash using `password_service.hash_password(secrets.token_urlsafe(32))` and then verified it every time an invalid username was submitted.
**Learning:** While the intent was to prevent user enumeration via timing attacks (by making the server take a consistent amount of time), dynamically generating the hash *before* verifying it caused the server to execute the CPU-intensive Argon2 algorithm twice. This introduces a severe Denial of Service (DoS) vulnerability where attackers can trivially exhaust server CPU by requesting invalid usernames.
**Prevention:** To prevent both timing attacks and DoS attacks, the application must use a *pre-computed* static dummy hash generated once at application startup. This ensures the server burns the correct amount of CPU during the verification step, without the penalty of generating a new hash.
## 2026-04-15 - [Insecure PyTorch Deserialization]
**Vulnerability:** Arbitrary Code Execution (Bandit B614). The `torch.load()` function uses `pickle` under the hood. When loading an untrusted model file without restrictions, a maliciously crafted file could execute arbitrary code during deserialization. This was found in `src/ml/export_onnx.py` where a PyTorch model is loaded to export to ONNX.
**Learning:** Even when loading models specifically meant for export, standard deserialization risks apply. PyTorch offers a built-in safety flag to restrict the unpickler to only return primitive types, dictionaries, and tensors.
**Prevention:** Always use `weights_only=True` when calling `torch.load()` unless executing arbitrary Python code from the model payload is absolutely required and trusted.
