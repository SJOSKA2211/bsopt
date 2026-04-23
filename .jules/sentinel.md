## 2024-05-24 - Unauthenticated Mock Bypass in API
**Vulnerability:** A development mock bypass using the X-User-ID header allowed full authentication bypass without verifying the environment.
**Learning:** Mock bypasses left uncommented or without strict environment checks introduce critical authentication vulnerabilities in production.
**Prevention:** Always gate development-only mock logic behind strict environment checks like settings.ENVIRONMENT != 'production'.
