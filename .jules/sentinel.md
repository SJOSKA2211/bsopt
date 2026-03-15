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
## 2026-03-15 - [Sentinel: Fix sensitive token logging]
**Vulnerability:** The password reset and email verification generated link log messages were logging the actual token values in plain text, which could lead to account compromise if log files are leaked.
**Learning:** Hardcoded logs containing URLs with query string tokens bypass protections and could be logged by downstream collectors (like CloudWatch, Datadog).
**Prevention:** Never log the parameters or sensitive elements of URLs sent via email or generated for users. Modify logs to remove token query variables.
## 2026-03-15 - [Sentinel: Fix SQL Injection in data_loader.py]
**Vulnerability:** The `fetch_system_metrics` method in `src/ml/data_loader.py` was using an f-string to insert the `hours` and `self.limit` variables directly into a SQL query. This exposes the database to SQL injection attacks if these variables are ever influenced by user input.
**Learning:** Even internal or admin-focused data loaders must use parameterized queries when fetching data from the database. Bandit caught this with warning B608 (hardcoded SQL expressions).
**Prevention:** Use native parameterized queries with placeholders like `$1` and `$2` when using `asyncpg`, and pass the parameters as separate arguments to `conn.fetch(query, arg1, arg2)`.
## 2026-03-15 - [Sentinel: Fix duplicate Cargo dependency]
**Vulnerability:** `src/core/Cargo.toml` contained a duplicate `num-complex = "0.4"` entry. While not a direct security vulnerability, it broke the Rust build via `maturin`, preventing security tests and deployments from succeeding in CI.
**Learning:** Broken builds mask true security issues by preventing CI pipelines from completing their scan stages. Keeping build definitions clean is a prerequisite for security enforcement.
**Prevention:** Ensure dependencies are not duplicated when manually adding libraries to `Cargo.toml`, or use `cargo add` which automatically handles deduplication.
## 2026-03-15 - [Sentinel: Fix PyJWT version in pyproject.toml]
**Vulnerability:** The `pyproject.toml` file pinned `PyJWT==2.8.0`. PyJWT < 2.12.0 accepts unknown `crit` header extensions which is a violation of RFC 7515 and poses a security risk allowing malicious tokens to bypass restrictions.
**Learning:** Hard-pinning older dependencies blocks security patches. Trivy scan flagged this CVE (CVE-2026-32597).
**Prevention:** Update `pyproject.toml` dependencies to safely rely on `>=` limits for security patches, rather than strict `==` versions unless necessary.
