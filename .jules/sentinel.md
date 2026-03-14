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

## 2024-03-14 - Secure PostgreSQL Variable Configuration
**Vulnerability:** PostgreSQL does not natively support parameter binding for the `SET` command. Using parameters like `SET LOCAL app.current_user_id = :user_id` throws an execution error. Developers may resort to unsafe string concatenation (f-strings) to work around this, leading to potential SQL injection.
**Learning:** `set_config` is a native PostgreSQL function that safely sets configuration variables and fully supports parameter binding, unlike the `SET` command.
**Prevention:** Use `SELECT set_config('variable_name', :value, true)` instead of `SET LOCAL` for setting session-level PostgreSQL variables safely in SQLAlchemy.

## 2024-03-14 - Fix Hardcoded SQL Expressions in Asyncpg
**Vulnerability:** A hardcoded SQL expression (Bandit B608) was found in `AIOpsDataLoader` where `hours` and `self.limit` were interpolated via f-strings into a raw SQL query. While `hours` and `self.limit` are generally integers, using string formatting for SQL queries introduces an unnecessary risk of SQL injection if the types were to change or input validation failed.
**Learning:** `asyncpg` strictly requires parameterized queries via positional arguments (`$1`, `$2`) to prevent SQL injection. For interval strings, `INTERVAL '{hours} hours'` can be safely parameterized using the PostgreSQL concatenation operator `||` alongside casting, like `($1 || ' hours')::interval`.
**Prevention:** Always use parameterized `$1, $2` syntax when calling `asyncpg`'s `conn.fetch(query, *args)` instead of f-strings, even for internal or integer-only values, to appease static analysis tools and enforce defense in depth.
