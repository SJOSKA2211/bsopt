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

## 2024-03-05 - [Insecure Deserialization in ML Pipeline]
**Vulnerability:** The machine learning pipeline in `src/ml/reinforcement_learning/offline_train.py` and `src/ml/distributed_training.py` used the insecure `pickle.load` for loading trajectory data. This could allow an attacker to execute arbitrary code (RCE) via insecure deserialization if they controlled the `.pkl` files.
**Learning:** `pickle` allows arbitrary object instantiation during deserialization and is inherently unsafe for untrusted data. Fallback mechanisms in data-loading logic often rely on legacy formats like `.pkl` out of convenience, bypassing newer, safer formats like `.parquet` or `.json`.
**Prevention:** Never use `pickle` for deserializing data that could come from untrusted sources. Migrate to safe serialization formats like JSON, Parquet, or Protobuf. Always enforce strict typing and bounds checking when loading external datasets.
