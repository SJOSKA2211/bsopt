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

## 2026-04-15 - [Insecure Deserialization in ML Pipeline]
**Vulnerability:** The ML training pipeline (`src/ml/reinforcement_learning/offline_train.py` and `src/ml/distributed_training.py`) used insecure `pickle.load()` as a fallback mechanism for loading trajectory datasets if Ray Data loading or `.parquet` loading failed. This allows Arbitrary Code Execution (ACE) if an attacker can supply a malicious `.pkl` file.
**Learning:** Legacy ML fallback logic often retains insecure practices (like reading `.pkl` files) to maintain backwards compatibility, even after primary paths migrate to secure formats like `.parquet`. Adding `# nosec B301` to suppress Bandit warnings for `pickle.load()` masks the vulnerability without fixing it.
**Prevention:** Always enforce secure deserialization formats (e.g., Parquet, SafeTensors) universally across primary and fallback paths. Remove legacy `.pkl` fallbacks entirely, explicitly validating file extensions and raising errors for unsafe formats.
