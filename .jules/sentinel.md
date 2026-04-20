## 2026-04-20 - Fix Authentication Bypass
**Vulnerability:** A mock development authentication bypass using the X-User-ID header was left active in production environments.
**Learning:** Development mock functions or bypasses left in production code lead to critical vulnerabilities, such as unauthorized access and privilege escalation.
**Prevention:** Always gate mock bypasses and testing functions with strict environment checks (e.g. settings.ENVIRONMENT != 'production').
