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
## 2026-03-07 - [Authentication Bypass via Insecure JWT Fallbacks]
**Vulnerability:** The `OIDCProvider.verify()` method in `src/auth/providers.py` had a critical security flaw where, if dynamic JWKS verification failed or no key was found, it fell back to accepting the token without any signature verification (`options={"verify_signature": False}`) or using a hardcoded, insecure secret (`key="secret"`). This allowed any attacker to forge JWTs and completely bypass authentication.
**Learning:** Legacy or "POC/Test" code paths left in production authentication logic create critical vulnerabilities. Fallback logic in security components must fail securely (e.g., throwing an exception) rather than failing open or using weak credentials.
**Prevention:** Never use `verify_signature: False` or hardcoded secrets in production code. Authentication providers must strictly enforce signature verification. Ensure test environments use properly mocked JWKS rather than disabling security checks in the core application code.
